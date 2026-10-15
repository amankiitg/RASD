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

## STEP 1: measurements

Two 8x sessions on `gpu_8x_a100_80gb_sxm4`, both in `us-east-1`, $22.32/h.

| | session | instance | rows landed | wall | cost |
|---|---|---|---|---|---|
| phase 1 | 2026-10-15 11:44Z | `e89726eed155444483aa9c613775bef5` | bench both arms + 6 blocking engine rows + 1 async row | 1860 s | $8.62 |
| phase 2 | 2026-10-15 12:20Z | `208d4e02c87747cab9f6d8989bee3d21` | 5 more engine rows (incl. 1M and async spec) | 3604 s | $22.34 |
| 1x rehearsal | 2026-10-15 10:5x | n/a (1x A100) | plumbing only, no measurements | 240 s | $0.37 |

Row provenance: every phase-1 row records `git_commit=31724c973fea`, every phase-2 row
`git_commit=135f8fb87d4a`. Evidence is committed at `results/mlsys/bw_diag_p1/` and
`results/mlsys/bw_diag_p2/`: the row CSVs (which carry `env_arm`, `nccl_env` and
`tokens_sha256` per row), the bench CSVs, `token_digests.json`, `session_meta.json`,
`topo.txt`, and a trimmed form of every stage log (stage lines, tracebacks and error markers
only, each line truncated). The token sidecars carry the full 128k and 1M prompt id lists at
about 1 MB and 8.7 MB each, so they are not committed; the generated ids, round counts and
accept counts from all 15 of them are extracted into
`results/mlsys/bw_diag_token_streams.json` (12 KB).

**Deviation from the brief, stated up front.** The brief asked for one 8x session with a $30
ceiling. Phase 1 could not answer the brief's own questions: the async arm's spec row and its
1M row both failed, so "ring vs splitkv under Arm B" and "TTFT under Arm B" had no data. I ran
the second session with a $20 session cap and a 75-minute timer to fill exactly those gaps. It
came in at $22.34, so 8x spend for this run is $30.96 against the $30 ceiling ($31.33 including
the 1x rehearsal). That is $1.33 over, and it is a real overrun of the stated ceiling; no
further spend will be made. Cumulative project spend is now about $908 of the $1,000 cap.

### 1a. Bare ring microbench: the decisive measurement

`scripts/mlsys_ring_microbench.py`, 8 ranks, the same `batch_isend_irecv` ring primitive the
engine uses, 256 rounds, rank 0, both timing modes (0.5), both arms, same box, same session:

| payload | A `wait` | B `wait` | A `sync` | B `sync` | B `sync` GB/s |
|---|---|---|---|---|---|
| 4 KB | 10.3347 ms | 0.1039 ms | 0.1487 ms | **0.1340 ms** | 0.03 |
| 33 MB | 10.3398 | 0.1017 | 0.8373 | 0.8135 | 40.93 |
| 133 MB | 10.3442 | 0.0997 | 2.5420 | 2.5306 | 52.62 |
| 266 MB | 10.3520 | 0.1045 | 2.6404 | **2.6191** | **101.69** |
| 66 MB (8 heads, expand-after) | 10.3802 | 0.0969 | 1.4371 | 1.3633 | 48.84 |

Read the `sync` columns: they are the same in both arms to within 3% at every payload, because
the wait mode changes when the host waits, not what the wire does. Arm A's `wait` is flat at
10.33 to 10.38 ms from 4 KB to 266 MB: that is the 10 ms sleep plus the check, and it is
independent of payload. Arm B's `sync` at 4 KB is 0.134 ms, which is 77x below Arm A's round,
and it rises monotonically with payload to 101.69 GB/s at 266 MB.

**The artefact is confirmed.** The flat ~10.3 ms was the host-side poll of
`TORCH_NCCL_BLOCKING_WAIT=1`, not NCCL and not NVLink. `nvidia-smi topo -m` (committed at
`results/mlsys/bw_diag_p1/topo.txt`) shows all eight GPUs connected NV12 to each other, so the
transport was never in question; the NIC is PHB-attached and unused.

### 1b. 128k engine rows (pg19_train_0, seed 42, greedy, same box, in-session)

16-token target-only rows and 32-token spec rows, decode-only = (time_sec - ttft) / (tokens - 1):

| engine | mode | A (blocking) | B (async) | A / B |
|---|---|---|---|---|
| ring2048 | target-only | 3.7805 (P1), 3.7950 (P2) | 2.0319 (P1), **1.9257 (P2)** | 1.86x, **1.97x** |
| ring32768 | target-only | 2.9553 (P1), 3.0746 (P2) | crash | n/a |
| splitkv | target-only | 1.0706 (P1), **1.0013 (P2)** | crash | n/a |
| ring2048 | spec | 2.8811 (P1), 2.8482 (P2) | **1.4052 (P2)** | **2.03x** |
| ring32768 | spec | 2.2914 (P1) | not reached | n/a |
| splitkv | spec | 0.9026 (P1) | not reached | n/a |

TTFT in the same rows: A 8.90 s (P1) and 12.39 s (P2) for the identical ring2048 target-only
config, B 9.86 s (P1) and 9.07 s (P2). A same-config, cross-session spread of 39% is larger
than any difference between the arms, so **TTFT is not resolvable in this harness**; see STEP 2
item 2. The 1M rows are the exception, where TTFT is 335 to 338 s and stable.

Reconciliation of the arm difference against the microbench, 128k ring2048 target-only:
224 rounds/token times (10.3347 - 2.6404) ms = 1.724 s predicted; 3.7805 - 2.0319 = 1.749 s
measured (P1), a 1.4% discrepancy. The poll is therefore 46 to 49% of the ring's decode step
under Arm A, matching the phase-1 note.

### 1c. 1M rows (pg19_train_0, seed 42)

| engine | mode | A (blocking) | B (async) | A / B |
|---|---|---|---|---|
| ring2048 | target-only | 16.4348 s/token, TTFT 338.45 s | 15.9613 s/token, TTFT 334.83 s | 1.03x |
| ring2048 | spec | 7.5637 s/token, TTFT 332.56 s | crash | n/a |

At 1M the flag is worth 2.9% on decode and 1.1% on TTFT, against 97% on decode at 128k. The
poll model predicts about 1.1 s of the 16.4 s step (224 rounds times a mean half-sleep), so the
measured 0.47 s is 2.4x smaller than predicted; the single row per arm cannot separate that
from run-to-run noise in a 16 s step, and it is reported as unresolved in STEP 2 item 3.

### 1d. Failures, and the secondary finding

Arm A landed 11 engine rows across the two sessions with zero crashes, five of them repeat
cells, which is how the blocking spec digest below is shown to reproduce across sessions and
boxes. Arm B landed 4 rows and **crashed on three of its five stages**, always with
`RuntimeError: CUDA error: device-side assert triggered` and, in the rank traceback, an
embedding `indexSelectLargeIndex` assert.

| phase | stage | cell being run when it died |
|---|---|---|
| 1 | `B[async]` arms | ring2048 spec @128k (after ring2048 target-only succeeded) |
| 2 | `B1[async]` target-only | ring32768 target-only @128k (after ring2048 target-only succeeded) |
| 2 | `C[async]` 1M | ring2048 target-only @1M |

Three different cells, so the hazard is not tied to a block size, an arm or a mode. In phase 1
the two 1M failures were a different, benign cause: `mlsys_df_arms.py` hard-coded the 128k-window
prompt pool, fixed with `--prompt-documents-json` (commit `135f8fb`); the phase-2 async 1M
failure is a genuine assert, because the phase-2 blocking 1M row succeeded on the same pool.

Peak reserved memory, worst rank, same cell:

| cell | A (blocking) | B (async) | change |
|---|---|---|---|
| ring2048 target-only @128k | 10,485.8 MB | 14,512.3 MB | +38% |
| ring2048 target-only @1M | 33,617.3 MB | 58,405.7 MB | **+74%** |

Without the CPU-side wait the host runs ahead of the device, so more buffers are live at once.
At 1M that is 58.4 GB of an 80 GB card.

### 1e. Token streams

`results/mlsys/bw_diag_token_streams.json` holds the id lists and round traces; the sha256 per
row is in each `df_arms.csv`. Cross-arm comparison, per engine and cell:

| cell | digest A | digest B | identical |
|---|---|---|---|
| ring2048 target-only @128k | `06c0b516452ad2ea` | `06c0b516452ad2ea` | yes (all 6 rows at 128k, both arms, both sessions, both block sizes) |
| ring32768 target-only @128k | `06c0b516452ad2ea` | crash | yes where it exists, and equal to ring2048 |
| ring2048 target-only @1M | `2e3bcaa241adb14e` | `2e3bcaa241adb14e` | yes |
| splitkv target-only @128k | `e01a287093ff0a62` | crash | differs from ring by design (reduction order) |
| ring2048 spec @128k | `19dc491d2a3033b1` | `438cfc0381057940` | **NO** |
| ring32768 spec @128k | `19dc491d2a3033b1` | not reached | equal to the ring2048 spec stream |
| splitkv spec @128k | `a4a6a314fda8cd20` | not reached | differs from ring by design |
| ring2048 spec @1M | `50a3a63a0ce4706a` | crash | n/a |

The blocking digest for the 128k ring2048 spec cell is identical in phase 1 and phase 2, so the
blocking arm is reproducible across sessions and boxes. The async arm produced a different
stream in the one cell where both arms completed: divergence at token index 8, with 13 accepted
drafts in 19 rounds against 8 in 23. These are greedy, seeded, `ignore_eos` rows against the
same prompt on the same box, so a difference is not sampling noise.

## STEP 2: analysis

### 2.1 Is the artefact confirmed?

Yes, and the intended test passes: the Arm B 4 KB round is 0.1340 ms, three orders of magnitude
below 1 ms and 77x below Arm A's 10.3347 ms. The flat line was the flag. The same primitive is
bandwidth-dominated above roughly 4 KB (0.134 ms to 2.619 ms as payload goes 4 KB to 266 MB).

### 2.2 Old against new, both arms

| quantity | A (blocking) | B (async) | ratio |
|---|---|---|---|
| ring round, 4 KB (host-observed) | 10.3347 ms | 0.1340 ms | 77x |
| ring round, 266 MB | 10.3520 ms | 2.6191 ms | 3.95x |
| ring2048 decode, target-only @128k | 3.7950 s/token | 1.9257 s/token | 1.97x |
| ring2048 decode, spec @128k | 2.8482 s/token | 1.4052 s/token | 2.03x |
| splitkv decode, target-only @128k | 1.0013 s/token | not measured | n/a |
| ring2048 decode, target-only @1M | 16.4348 s/token | 15.9613 s/token | 1.03x |
| ring2048 decode, spec @1M | 7.5637 s/token | not measured | n/a |
| ring2048 TTFT @128k | 8.90 to 12.39 s | 9.07 to 9.86 s | not resolvable |
| ring2048 TTFT @1M | 338.45 s | 334.83 s | 1.01x |

TTFT deserves its own line because the brief asked for it: the identical ring2048 target-only
config measured 8.90 s in phase 1 and 12.39 s in phase 2 under Arm A, and 9.86 s and 9.07 s
under Arm B. Phase 1 says Arm B is 1.0 s slower, phase 2 says it is 3.3 s faster. Prefill does
use the ring, so an effect is plausible in either direction, but this instrument cannot measure
it. Any TTFT-based claim about the flag, including the vLLM prefill ratio, needs a cleaner
timing before it is repeated.

### 2.3 Token streams: the stop condition fired

Target-only rows are bit-identical across arms at both contexts, and the blocking spec stream
reproduces across sessions. **The one spec cell measured in both arms diverges at token index
8.** That is the brief's second stop condition, and it is reported rather than worked around: I
did not rerun the cell to look for a friendlier sample, and no further 8x spend was made.

The crashers in 1d point the same way. The likely mechanism is the one already suspected in the
decode diagnosis: the ring's per-token rotation swaps and rewrites `k_cur`/`k_buf` between
collectives, and under Arm A the host's 10 ms sleep per round guarantees the collective has
drained before the host enqueues the next write into that buffer. Removing the CPU-side wait
removes that accidental ordering barrier, which both corrupts activations (the embedding index
assert) and lets the host run far enough ahead to inflate live buffers by 38 to 74%. The
divergent spec stream and the asserts are therefore one finding, not three: **the flag was
load-bearing for correctness, not only for speed.**

### 2.4 What fraction of the ring's decode step is really communication

Under Arm B the ring's 128k decode step is 1.9257 s/token and the wire is 224 rounds times
2.6191 ms = 0.5867 s, so **30.5% is communication and 69.5% is the rotation, elementwise work
and attention compute**. Under Arm A the same step is 3.7950 s, of which 0.5867 s is wire,
1.7486 s is the poll (46.1%) and 1.4597 s is the rest. The honest description is therefore the
opposite of F4's: the ring's decode step under the shipped config was dominated by a host-side
sleep, and once that is removed the ring is *not* latency-bound at 128k, it is 70% local work.

### 2.5 What split-KV is worth under Arm B

Split-KV under Arm A at 128k target-only is 1.0013 s/token against the ring's 3.7950, a 3.79x
speedup. Under Arm B that cell was not measured (the async stage died on the row before it), so
the honest answer is a bracket derived from the measured per-op poll cost:

- split-KV decode merges `(out, lse)` across ranks once per layer, 32 layers, so 32 to 64
  collectives per token against the ring's 224 rounds. At 10.35 ms per in-flight collective
  that is 0.33 to 0.66 s of poll.
- Removing it leaves roughly 0.34 to 0.67 s/token, so against the ring's 1.9257 s under Arm B
  split-KV is worth **about 2.9x to 5.7x**, and no less than 1.92x (it cannot be slower than
  its Arm A number).

The direction and even the magnitude of F4's headline range (2.80x to 5.32x) survive under Arm
B, but for a different reason: under Arm A both arms paid the poll, so the ratio was never the
measurement of the ring's cost that it was presented as. The mechanism sentence was wrong; the
conclusion that split-KV is the fix is still right.

### 2.6 Split-KV's own timing is also polluted

Worth recording because it is a paper-internal consequence: the Arm A splitkv rows carry the
same 10 ms-per-collective handicap as the ring. The 32 to 64 merges per token cost 0.33 to
0.66 s of the 1.0013 s the splitkv target-only row reports, so at least a third of what that
row measures is the flag. Any split-KV absolute number in the paper is an Arm A number.

### 2.7 Projected cost to rerun the headline table under Arm B

Headline cells per `CAMERA_READY_KIT.md`: 128k spec n=6 and target n=4, 512k spec n=3 and
target n=3, 1M spec n=15 and target n=5, so 36 rows. Cost model from the measured stage walls:

| component | measured | note |
|---|---|---|
| 128k target-only row under B | 38.0 s | 65.6 and 69.3 s under A |
| 128k spec row under B | 54.6 s | 96.8 and 99.6 s under A |
| 1M target-only row under B | 574.3 s | 585.0 s under A |
| non-row stage time @128k | 30 s | prompt build and load, phase 2 stage B1 |
| non-row stage time @1M | 624 s | stage wall minus row time, phase 2 stage C |

| context | rows | under A | under B | saving |
|---|---|---|---|---|
| 128k | 10 | $4.4 | $3.7 | $0.7 |
| 512k | 6 | $16.4 | $15.1 | $1.3 |
| 1M | 20 | $149.0 | $148.1 | $0.9 |
| total | 36 | **$169.8** | **$166.9** | **$2.9 (1.7%)** |

512k row time is projected from the measured 512k target-only decode (2.1480 s/token under A)
with the 128k arm ratio, and its non-row time is interpolated between the two measured values;
those are the two soft numbers in the table, and they do not change the conclusion. The headline
table is dominated by the 1M rung, where the flag is worth only 1 to 3% because a 1M decode step
is 16 s and prefill plus prompt construction is 625 s per row. Removing the flag would buy about
$3 on a $170 table, which is inside the noise of this model, and $170 is above the $92 of
headroom left under the $1,000 cap in any case, so the rerun is not worth doing and is not
proposed.

### 2.8 Which claims survive

| claim | verdict under Arm B |
|---|---|
| F1 RoPE intervention (acceptance and ppl) | **survives.** Both are token-level quantities on the target or the two models, and the target-only stream is provably unchanged by the arm. The three lifted speedups (3.36x / 2.84x / 2.76x, n=1 target-only) were measured under A; keep the kit's recommendation to drop or label them, and note they are Arm A numbers. |
| F2 greedy degeneracy | **survives.** Token classification on target-only rows, whose streams are bit-identical across arms. |
| F3 1M coherence and ppl | **survives.** ppl comes from prefill, and the 1M target-only stream is bit-identical across arms. |
| F4 ring decode diagnosis | **mechanism refuted; fix survives.** The 10.3 ms flat round was the flag. Under B the same primitive is 0.134 ms at 4 KB and 101.7 GB/s at 266 MB, and the ring's decode step is 30.5% wire. The "fixed per-round latency, not bandwidth" sentence and any extrapolation that cost scales with rank count and not bytes are casualties. Split-KV is still worth roughly 2.9x to 5.7x at 128k, so the headline range survives; every absolute number in it is an Arm A number. |
| F5 spec on top of split-KV | **direction survives, ratio needs re-measuring.** At the one cell measured in both arms the spec-over-target ratio is 1.37x under B against 1.33x under A, so removing the poll does not destroy the speculative advantage. But those rows sit at acceptance 0.087 to 0.153, far below the paper's 128k cells, so they are not the paper's ratio. The paper's 2 to 2.5x was measured under A. |
| vLLM prefill (ring 2.0x to 2.5x faster) | **not resolvable; re-measure before repeating.** TTFT in this harness varies 39% between sessions for an identical config, and the two arms disagree in sign. Prefill does pay the poll, so an effect exists, but this instrument cannot size it. |
| vLLM decode | **must be restated per context and per arm.** RASD's decode improves 1.97x at 128k under B but 1.03x at 1M, so a single overall decode gap is wrong in both directions. The precision mismatch (vLLM bf16 against FP4 weights and NF4 KV) belongs in the same sentence as every ratio, as the kit already requires. |

### 2.9 Recommendation

Do **not** adopt Arm B as a drop-in replacement on the strength of this run, and do not change
the campaign's pod env. The flag is worth 2x at 128k, which is a lot, but it is currently
carrying correctness: with it removed, the only spec cell measured in both arms diverges from
token 8, three of five async stages died on a device-side assert, and peak memory rises by up to
74% at 1M. The order of work if the poll is to go is:

1. fix the ring's buffer-reuse ordering so it does not depend on the host having drained the
   collective (the rotation swap is the prime suspect), and
2. re-verify bit-identity on spec rows in both arms, at ring2048 and ring32768, before any
   measurement is taken under Arm B, and
3. only then re-time, with a TTFT instrument that is not the current one.

What is already true and worth committing regardless: `MLSYS_NCCL_WAIT_MODE` exists, exactly one
wait line is written to the pod env, every new row records `env_arm` and `nccl_env` (including
`<unset>`), and the campaign's measurements can now say which configuration produced them. That
is a provenance improvement that does not depend on the answer to the artefact question.

### STOP

Stop condition 2 (Arm A and Arm B token streams differ) has fired on the 128k ring2048 spec
cell. I have stopped: no rerun of the cell, no headline table, no paper edit. Every instance for
both phases was terminated and confirmed terminated at the end of its session, no dead-man timer
or capacity poller is running, and 0 cloud instances are held.
