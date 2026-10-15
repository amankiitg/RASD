# TORCH_NCCL_BLOCKING_WAIT: diagnostic

Branch `diag/blocking-wait`, from tag `lcfm-camera-ready-v1` (commit `c0fa9c7`). Diagnostic
only: nothing under `manuscript/` is touched. An independent review (CODEX_REVIEW.md, item B2)
claims that the flat ~10.3 ms per ring round is a host-side polling artefact of
`TORCH_NCCL_BLOCKING_WAIT=1` rather than NCCL or NVLink latency, and that F4's diagnosis
therefore describes a configuration flag. This run tests that claim directly.

Status: STEP 0 complete (below). STEP 1 and STEP 2 follow in this file.

---

## STEP 0: where it is set, what it was protecting, and the replacement

### 0.1 Every place the variable is set

| file | line | what it does |
|---|---|---|
| `scripts/mlsys_pod_env.sh` | 32-35 | builds `NCCL_SETTINGS` containing `TORCH_NCCL_BLOCKING_WAIT=1` |
| `scripts/mlsys_pod_env.sh` | 153-154 | writes that block, `export`-prefixed, to `$REPO/.pod_env.sh` |
| `scripts/mlsys_pod_session.sh` | 73 | older standalone flow, same export |
| `scripts/phase_c_pod_session.sh` | 53 | Phase C flow, same export |
| `scripts/phase_c_pod_session_40gb.sh` | 42 | Phase C 40 GB variant, same export |
| `scripts/r6_async_debug.sh` | 22 | async-debugging helper |
| `scripts/r6_verify_runner.sh` | 26 | R6 verification runner |
| `docs/pod_setup.md` | 112 | the operator prose, "NCCL and allocator settings" |
| `docs/dev/PHASE_C_RUNBOOK.md` | 199 | the runbook copy |

The **campaign path** is the first one. Every campaign pod runs
`bash scripts/mlsys_pod_env.sh` during provisioning (driven by `scripts/mlsys_session_a.sh`),
which writes `~/RASD/.pod_env.sh`. Both remote scripts then source it with `set -a`:

* `scripts/mlsys_session_a_remote.sh:62` (`set -a; . "$HOME/RASD/.pod_env.sh"; set +a`)
* `scripts/mlsys_df_remote.sh:81` (same line, the diagnostic arms and the ring bench)

**This is why the variable was not per-run controllable.** `set -a` means the sourced file's
assignments *override* anything the driver forwarded through `FORWARD_ENV`, so a forwarded
value could be silently overwritten on the pod. Nothing in the row recorded which mode a run
used either: `_NCCL_KEYS` in `run_experiment.py` (line 223) lists 16 NCCL knobs and none of
them is the wait mode or any of the `TORCH_NCCL_*` timeout variables. A row therefore could
not prove its own wait mode, which is the same class of defect as the missing engine
provenance the earlier run fixed.

### 0.2 What it was protecting against, and why it was added

It was added on 2026-05-10 in commit `f1a448a` ("Bump NCCL watchdog to 1 hr"), together with
`NCCL_TIMEOUT=3600` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`, and the commit message says
exactly what for: a 1M smoke test had died not from OOM but from the **default torch NCCL
watchdog of 10 minutes** firing on a coalesced ring-attention collective
(`WorkNCCL(SeqNum=36, OpType=COALESCED, ...) ran for 600075 ms`). The fix was
`dist.init_process_group(timeout=timedelta(hours=1))` in code, plus the three env vars as a
safety net "since some torch versions ignore the kwarg".

So the intent was **hang detection with a long enough budget**, at a time when 1M prefill
takes tens of minutes. It worked as a detector: the 2026-10-09 divergence left
`timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1)` in
`results/mlsys/incident_20261009T001856Z/pod_results/pod_home/manifest.log:200`. That log
line is also independent evidence that the flag was live in the campaign pods.

### 0.3 The mechanism, quoted from the pinned torch

The pod image pins `torch==2.5.1+cu124` on python 3.10 (`requirements-lock.txt:114`, and
`docs/pod_setup.md` uses the same pin for the flash-attn wheel tag). From
`pytorch/pytorch` tag `v2.5.1`, `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`:

```cpp
constexpr int64_t kSynchronizeBusyWaitMillis = 10;            // line 426

void ProcessGroupNCCL::WorkNCCL::synchronizeInternal(std::chrono::milliseconds timeout) {
  synchronizeStream();
  if (blockingWait_) {                                        // line 664
    while (!isCompleted()) {                                  // line 665
      bool timedOut = checkTimeout(...);                      // line 666
      ...
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));  // lines 683-684
    }
  }
  ...
}
```

`isCompleted()` (line 514) is `exception() || finishedGPUExecutionInternal()`, and the latter
queries a CUDA event (`ncclEndEvent_->query()`, line 584), which is **non-blocking**. So while
a transfer is in flight the CPU sleeps a fixed **10 ms** and re-polls. `synchronizeStream()`
(line 648) only *enqueues* a stream dependency (`ncclEndEvent_->block(currentStream)`); it
does not wait.

Two consequences that decide the experimental design:

1. **The mode is fixed per process group.** `blockingWait_ = getCvarBool(TORCH_NCCL_BLOCKING_WAIT, false)`
   is read once in the `ProcessGroupNCCL` constructor (line 826) and copied into every work
   (`work->blockingWait_ = blockingWait_`, lines 2618/2814/2993/3220). Arm A and Arm B are
   therefore **separate process launches**; no in-process switch exists.
2. **In Arm B a `wait()` on a point-to-point work does almost nothing.** With
   `blockingWait_` false, `synchronizeInternal` runs `synchronizeStream()` and then, for works
   with no `barrierTensor_` (all P2P sends and receives), returns (lines 696-715). Correctness
   is preserved by the *stream* dependency, not by a CPU wait. So if the microbench keeps
   timing `r.wait()`, Arm B would measure **enqueue cost, not transport**. That is why the
   microbench gains a second timing mode (0.5) that forces a real device synchronisation inside
   the timed region.

One further fact worth recording, because it is a second cost of the flag: when
`blockingWait_` is true and async error handling is not `NoHandling`, torch **disables** async
error handling and says so:

```cpp
if (blockingWait_) {
  if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
    LOG(INFO) << logPrefix() << "TORCH_NCCL_BLOCKING_WAIT and "
              << "TORCH_NCCL_ASYNC_ERROR_HANDLING|TORCH_NCCL_DESYNC_DEBUG"
              << "should not both be enabled. "
              << "Only TORCH_NCCL_BLOCKING_WAIT is being used in this process.";
    asyncErrorHandling_ = NoHandling;                         // lines 875-882
    desyncDebug_ = false;
  }
}
```

The default `TORCH_NCCL_ASYNC_ERROR_HANDLING` is 3 (`SkipCleanUp`, line 828), so this branch
always fired: **the campaign ran with async error handling off**, the exact mechanism the
flag was supposed to provide.

### 0.4 The replacement: keep hang protection, drop the poll

A mode variable, `MLSYS_NCCL_WAIT_MODE`, resolved on the pod into the env file, so the mode is
named once and its *effect* is what the rows record:

| mode | what the pod exports | hang protection |
|---|---|---|
| `async` (new default) | `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and **no** `TORCH_NCCL_BLOCKING_WAIT` line at all | the NCCL watchdog thread (100 ms tick, `kWatchdogThreadSleepMillis`) plus `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`, tearing the process group down on a stuck collective; plus the process-group timeout from `dist.init_process_group` (`MLSYS_PG_TIMEOUT_MINUTES`, default 15) |
| `blocking` (historical) | `TORCH_NCCL_BLOCKING_WAIT=1` | as before: the 10 ms poll loop enforces the timeout in the wait path |

`NCCL_TIMEOUT=3600` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` are kept in both modes, so the
long 1M prefill still has its hour. The honest trade, stated rather than glossed: `blocking`
names the op and the sequence number in the exception it raises in the wait path, whereas
`async` reports through the watchdog and dumps on timeout instead. Both abort; the difference
is the quality of the message. Given that the poll costs ~10 ms per collective during
*normal* operation, and 224 of those per decode token, the trade is heavily one-sided.

Per-run control, and per-row proof:

* `MLSYS_NCCL_WAIT_MODE` is added to `FORWARD_ENV` and to the pod's `REMOTE_EXPECT` contract,
  so an unset or unarrived mode stops the launch before an instance exists (rule R9).
* `_NCCL_KEYS` in `run_experiment.py` gains the four wait and timeout keys, and a new
  always-emitted block appends the wait mode to the `nccl_env` column of **every** row with
  `KEY=<unset>` where a key is absent, so "unset" is recorded rather than merely implied.
* The arms driver gains an `env_arm` column (the intended arm) beside `nccl_env` (the effect),
  so a row that cannot prove its arm is void, exactly as the engine-provenance rule works.
* The remote script sets the variables per stage launch with `env` and `env -u` *after*
  sourcing `.pod_env.sh`, which is what makes two arms possible inside one session.

### 0.4a A finding from the 1x rehearsal: torchrun injects the async variable

The plan above assumed the two arms could each set exactly one wait knob. The 1x plumbing
rehearsal showed that is **impossible under `torchrun`**, and the reason is worth recording
because it also tells us what every past campaign run actually did.

`torch/distributed/elastic/agent/server/local_elastic_agent.py` builds each worker's
environment with an unconditional entry (identical in v2.5.1 and in later releases):

```python
"TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)),
```

So the elastic agent **defaults that variable to 1 in every worker**, whether or not the
parent set it, and `env -u` before `torchrun` cannot prevent it. The rehearsal is the proof:
the blocking arm was launched as
`env -u TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_BLOCKING_WAIT=1 ...` and its rows still
recorded `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, with `wait_mode=blocking`.

Two consequences, and the first is evidence rather than an inconvenience:

1. **Every campaign run had both variables set.** Every stage in this project was launched
   through `torchrun`, so every one of them had `TORCH_NCCL_BLOCKING_WAIT=1` *and*
   `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. From 0.3, torch resolves that combination by using
   blocking wait and **disabling** async error handling, logging
   `Only TORCH_NCCL_BLOCKING_WAIT is being used in this process`. The poll loop was therefore
   live in every run, which is what the artifact hypothesis requires, and the campaign never
   had the async path at all.
2. **The arms are distinguished by `TORCH_NCCL_BLOCKING_WAIT` alone**, so that is what the
   remote script sets or clears: `blocking` sets it, `async` clears it with `env -u` and lets
   the injected `ASYNC=1` supply error handling. `_nccl_wait_mode()` in `run_experiment.py`
   applies torch's own precedence (blocking first), so a row's recorded mode is the mode the
   process group actually used, not the intended one.

The test that covers this asserts the injection is *present* in the blocking arm rather than
absent, precisely because demanding its absence would be testing a fiction.

### 0.5 The measurement design (and the trap it avoids)

| stage | instrument | arm A (`blocking`) | arm B (`async`) |
|---|---|---|---|
| 1a | `mlsys_ring_microbench.py --timing wait` | what the engine's wait loop pays per round | enqueue cost only (see 0.3.2) |
| 1a | `mlsys_ring_microbench.py --timing sync` | the same, since the wait already blocked | **true transport time**, the number the claim is about |
| 1b | `mlsys_df_arms.py --mode arms`, ring2048 / ring32768 / splitkv, target-only and spec | decode s/token, TTFT, token sha256 | same |
| 1c | one 1M target-only row per arm | TTFT and decode at full context | same |

The `sync` mode exists because the claim under test is about *transport*, and Arm B's `wait()`
by construction does not wait. Reporting only `wait` in Arm B would show a dramatic speedup
that is partly an artefact of measuring a no-op, which would be the same class of error as the
one being investigated. Both numbers are reported for both arms.

Two further controls: the token-stream sha256 per row (Arm A and Arm B must agree per engine
arm, because the flag changes waiting and not arithmetic), and a 1x plumbing rehearsal of the
whole chain before any 8x spend.
