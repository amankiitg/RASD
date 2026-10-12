# HANDOFF — read this first

Written 2026-10-12T01:50Z at a clean boundary, on operator instruction. **Not**
`MORNING_REPORT.txt`: a running driver's puller owns that file (see lesson L2).

---

## 1. The session that is running right now — do not touch it

| item | value |
|---|---|
| instance id | **cad0d8cdaf2e4994ae7a70322ef1efe3** (`gpu_8x_a100_80gb_sxm4`, us-east-1, $22.32/hr) |
| launched | 2026-10-12T00:11:06Z |
| driver pid | **69660** (`bash scripts/mlsys_session_a.sh`, PPID 1, own process group) |
| timer pid | **74459** (`bash scripts/mlsys_deadman.sh`, PPID 1) |
| timer armed | 00:11:11Z for 480 min → **hard deadline 2026-10-12T08:11:11Z** |
| driver poll deadline | **2026-10-12T07:56:06Z** (timer deadline less 15 min) |
| expected remote finish | ~06:45Z by estimate from measured row times; the two deadlines above are the guarantees |
| config | `configs/mlsys_followup_sampling.yml` (protocol A, T=0.7), `SESSION_TAG=followup_samp2` |
| stage now | `SAMP_1M_SPEC` started 01:37:48Z (step d, 6 books) |
| progress | a sanity 2/2 ok → b QUALITY_FIX_128K 3/3, QUALITY_FIX_1M 3/3 → **c SAMP_CHECK_1M_SPEC PASSED** (acceptance 0.3776, 0.1068 tok/s, repetition-clean) → d running |

**Rules for this instance: do not edit any file it reads, do not edit `MORNING_REPORT.txt`
while the driver runs, and launch nothing else while it runs.**

Verify by effect before assuming any state:
```bash
ps -o pid,ppid,etime -p 69660 -p 74459        # both must exist, both PPID 1
curl -sS -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances | grep -c '"id"'
```

---

## 2. What Phase 1 produced (committed: `ed8957b`)

* `results/mlsys/BANDWIDTH_FLOOR.md` — the per-token traffic floor at 128k/512k/1M with all
  14 assumptions tabulated. **Floor 59/232/461 ms per token against measured 3.51/9.98/19.10
  s per token: a 42-59x gap.** Measured time is very nearly linear in the number of ring
  chunk transfers (4,096/16,384/32,768 per token at **0.58-0.86 ms each**), i.e. ~3.6 GB/s on
  a link capable of ~300 GB/s per direction. Bandwidth and attention FLOPs are both ruled
  out arithmetically; the cause candidates are quoted with code, including the
  `wait()`-per-request in `_ring_forward` that forbids overlap despite `prefetch_depth`.
* `results/mlsys/PROFILE_PLAN.md` — the two profiles, the exact commands, the decision rule,
  the caps, and an honest limitation (the runner's `--profile` wraps the whole `generate()`,
  so warm-up is at ROW granularity, not step granularity).
* `scripts/mlsys_profile_decode.sh` — one script for both profiles, `--nproc` parameterised so
  the two cannot drift; `--report-only` makes the reporting half testable with no GPU.
  Local testing found two real defects (wrong default interpreter; microsecond fields divided
  by seconds = a "99,166,666%" share). Both fixed; the synthetic case now reports 99.2%.
  **The run half cannot be tested on CPU: the engine calls `torch.cuda.set_device`.**
* `configs/mlsys_profile_128k.yml`, `configs/mlsys_profile_tiny_cpu.yml`.

---

## 3. Remaining work, exactly as specified

### PHASE 2 — when the session ends (trigger: the driver exits; expected ~06:45-07:56Z)
1. Confirm the driver pulled and terminated: **0 instances via the API, no driver or timer via
   `ps`**.
2. **Judge the session ONLY from the stage log lines and the stage CSVs, never from the DONE
   marker's rc** (lesson L1). Report every stage: rows ok/bad, skipped or not, and why. If
   step c produced no ok rows, report it as a FAILURE: the reason from
   `group_SAMP_CHECK_1M_SPEC.log`, whether the rerun pass retried it, and which stages were
   skipped.
3. **Close path A** in `scripts/mlsys_session_a_remote.sh`: drop the `[ "$prev" = "0" ]` guard
   so an empty check stage also exits 5. Add a test for the empty-stage case. Run the suite.

### PHASE 3 — profiling (after Phase 2)
4. Run profile (a) on one `gpu_1x_a100` (any 1x A100 type; in practice
   **`gpu_1x_a100_sxm4`**, because plain `gpu_1x_a100` has had zero capacity in every region
   all week), **60-minute dead-man timer via `scripts/mlsys_detach.py`, confirmed by PPID 1**,
   **at most $5**. Rehearsal rules apply: the script must have been tested locally first.
5. If (a) explains most of the measured per-token time, **stop there**. If not, run profile
   (b) on the cheapest available **2-GPU A100 SXM (NVLink)** instance, 60-minute timer, **at
   most $10**. If no such instance is available within 2 hours, **skip (b) and say so**.
6. Write `results/mlsys/PROFILE.md`: where the time goes as a breakdown that adds up to the
   measured per-token time; the gap to the bandwidth floor; what a fix would likely be and its
   expected effect. **Findings only — no engine changes.**

### PHASE 4 — write-up
10. Rewrite `results/mlsys/RESULTS_OCT.md`:
    * **Headline: the 1M sampling rows (T=0.7).** Per-row table with acceptance, tokens/s
      (**`tokens_generated / time_sec`, never the rounded field**), periodicity verdict (a
      nonzero period with low agreement and `first_periodic_token = 128` is **"no cycle"**),
      clean-only means with n, and speedups against the session's **mean target-only tok/s at
      that context**, stating that **target-only rows are subsampled**.
    * The greedy Oct 10-11 results as a **separate condition**, with the corrected headline
      (1M clean n=1: **1.61x**).
    * **Quality redo**: 1M vs 128k ppl per book (`ppl_tokens` 1024), alignment judged from
      `continuation_sha256` and the recorded offsets. Book 0 exactly aligned; for the others
      report the **overlap of scored tokens computed from the relative offset between arms**
      (e.g. a 92-token offset is about **91%**, not 98-99%). Frame it as *"the native-1M model
      stays coherent at 1M (within ~10% of 128k on the same tokens)"*, **not** "benefits from
      long context".
    * Aligned scaling (512k, 128k) and the RoPE section, with gate verdicts.
    * The bandwidth floor and the profiling breakdown, stated honestly.
    * Spend, hardware, and limitations (books per context, one seed, one temperature, sampling
      losslessness is distributional, speedups are against our own target-only path, no vLLM
      baseline).
    * Regenerate the scaling plot from **valid sampling-session CSVs only**.
11. Write `results/mlsys/CLAIMS.md`: one page listing each claim the paper can make, the exact
    evidence row(s), and each claim it must not make, with the reason.
12. Put a 10-line summary at the top of `MORNING_REPORT.txt` **only after no driver is
    running**: headline numbers, what profiling found, total spend, what is left to decide.
    Commit and push.

### PHASE 5 — vLLM prep (only after Phases 1-4 are done and committed)
1. Local, read-only first: read `scripts/mlsys_vllm_baseline.py` and the vLLM stage configs.
   Confirm the **write-before-mkdir** bug is fixed **and tested**. List everything else that
   would block a first run: vLLM version, install path, loading
   `gradientai/Llama-3-8B-Instruct-Gradient-1048k` (*rope_theta*, *max_position_embeddings*),
   tensor parallel 8, `max_model_len` for 128k/512k/1M, KV dtype (bf16), chunked prefill, and
   **how decode tok/s is measured**. Fix only script/config problems (**not `src/`**), with
   tests.
2. Memory estimate per GPU at 128k/512k/1M with TP=8 (weights, bf16 KV, activations, vLLM
   overhead). **Flag any context that will not fit.**
3. Write `results/mlsys/VLLM_PLAN.md`: the minimal comparison (target-only greedy decode tok/s,
   same model, same aligned prompts, 128k/512k/1M, 128 tokens, same 8x A100 80GB type), the
   exact commands, expected hours and cost, and how it will be reported next to RASD.
4. **1x REHEARSAL after profiling finishes**: one `gpu_1x_a100` (any 1x type), **90-minute
   timer**, **at most $10**. Install vLLM exactly as the 8x run would, load the Gradient 1048k
   model at 8k context, decode 128 tokens, report tok/s and whether it ran cleanly. Fix script
   problems, rerun once if needed. Terminate, confirm 0 instances.
5. Write a **GO / NO-GO for running vLLM on 8x tomorrow** into `VLLM_PLAN.md`, with the reason,
   estimated hours and cost. Commit and push. **Do not launch the 8x vLLM run.**

**Global limits throughout:** no new 8x launch and no vLLM 8x run in this plan; no `src/`
changes; never print keys or tokens; **cumulative spend stays under $450**; every paid
instance gets a detached dead-man timer confirmed by PPID 1 *before* work starts; every stop
is verified by effect; commit and push after each phase.

---

## 4. Read these first

1. `results/mlsys/HANDOFF.md` (this file)
2. `results/mlsys/MORNING_REPORT.txt` — the status block, and especially the **PAUSED** block
   with its resume checklist
3. `results/mlsys/RESULTS_OCT.md` — the current deliverable (greedy only, corrected headline)
4. `results/mlsys/BANDWIDTH_FLOOR.md`, `results/mlsys/PROFILE_PLAN.md`
5. `configs/mlsys_followup_sampling.yml` — the protocol A plan the running session executes
6. `scripts/mlsys_session_a.sh` (driver) and `scripts/mlsys_session_a_remote.sh` (pod side)
7. `scripts/mlsys_deadman.sh`, `scripts/mlsys_detach.py`
8. `scripts/mlsys_profile_decode.sh`, `configs/mlsys_profile_128k.yml`
9. `scripts/mlsys_vllm_baseline.py` and the vLLM stage configs (Phase 5)
10. `src/analysis/repetition.py` (the periodicity/repetition verdicts),
    `src/analysis/profiler.py` (what `--profile` writes), `src/models/nf4_dynamic_cache.py`
11. The pause capture in the session workspace:
    `pause_20261011T121926Z_followup_samp/` (full pod results tree + the driver's incremental
    snapshots)

---

## 5. Lessons from this week — each one has already cost something

**L1 — The DONE marker's rc is NOT a success signal.** If the check stage produces zero ok
rows, the verification block is skipped (its guard is `[ "$prev" = "0" ]`), the later stages
log `SKIP` and never run, the rerun pass retries the row *without* re-verifying it, and the
script exits **0**. A "clean" `rc=0` session can therefore have run none of d/e/f/g. Judge from
the **stage log lines and the stage CSVs**. Phase 2 item 3 closes this.

**L2 — The driver's puller WRITES `results/mlsys/MORNING_REPORT.txt`** (it folds the pod's
`progress.md` into it every 240 s). Never edit it while a driver runs; an earlier edit of mine
was silently clobbered this way. That is why this handoff is a separate file.

**L3 — No `src/` changes in this plan.** Any engine change invalidates every timing already
measured, including the rows the paper will quote. Report findings only.

**L4 — Launch everything long-lived through `scripts/mlsys_detach.py`, and verify by effect.**
The 2026-10-10 runtime restart killed a `nohup` watcher and left a healthy 8xA100 billing with
nothing alive that knew its id. Verify detachment by **PPID 1** (and, for instances, the API
count) — never by the command having been issued.

**L5 — `tokens_generated / time_sec`, never the rounded `throughput_tps` field.** The field
carries two significant figures (0.08 vs 0.08411), which moves the 1M speedup from 1.61x to
1.68x.

**L6 — Alignment comes from hashes and offsets, not from how a pool was sliced.** Use
`continuation_sha256` from the **token record** plus the recorded `prompt_tokens` /
`prompt_source_len`. Do **not** use the CSV's `generated_tokens_sha256`: it hashes the 8
*generated* tokens and differs between arms even when the scored continuation matches exactly,
so it would wrongly report every book as misaligned. The engine's window landing drifts:
`prompt_source_len` exceeds `prompt_tokens` by **0 / +9 / +17** at 128k and **0 / +101 / +65**
at 1M for books 0/1/915.

**L7 — The overlap correction for the quality rows.** Where the two arms did not score
identical tokens, the overlap must be computed from the **relative offset between the arms**
(a 92-token offset over a 1024-token window is ~**91%**), not from the "same book" intuition
that produced my earlier "98-99%". Only book 0 is exactly aligned.

**L8 — A dead-man timer can fail OPEN.** On 2026-10-11 the timer for a live, billing instance
logged "no longer exists" and exited, because it treated any single failed poll as proof the
instance was gone — one timeout, 5xx or empty body was indistinguishable from a real 404. It
left the mandated safety net off for about an hour. It now requires **three consecutive
authoritative answers (200/404)** and keeps waiting otherwise. The asymmetry decides the
design: a false "gone" costs an unbounded bill; staying armed costs nothing.

**L9 — An SSH return is not evidence that remote work ended.** The 2026-10-11 session was
terminated mid-stage because the Mac→pod ssh dropped and the driver treated that as completion
— losing the 128k target-only arm while the pod was still working. The remote script now runs
detached and writes its own `DONE`; the driver polls for that marker over short fresh ssh calls
and **keeps waiting when a poll fails**. This was proven by deliberately killing the poll ssh
mid-run: `POLL SSH FAILED (rc=137)`, driver alive, instance still active, polling resumed.

**L10 — Push out the branch: `gpu_1x_a100` has never had capacity; `gpu_1x_a100_sxm4` does.**
An exact-match poll on the wrong type waits until the cutoff for no reason.

**L11 — Rehearse on one 1x GPU with the EXACT command lines.** The rehearsal caught five real
defects before a paid 8× launch, including a gate that raised `NameError` for every arm, a
token file written before the directory it lived in existed, and a rope arm pointed at a model
that ships `rope_scaling: None` and cannot take the intervention at all.

**L12 — "Test the function" is not the same as "call the function".**
`generation_metrics` had **zero** test coverage, which is how the `NameError` shipped. A test
that merely calls it would have caught it.

**L13 — The silent-drop class of bug, three instances this week:** `token_topk` computed but
absent from the sidecar schema; a patch anchored on a string the script no longer contained;
a candidates file using `candidate` where the loader reads `name` (which loads **zero**
candidates and therefore looks exactly like a gate that passed). Assert on the artifact — the
committed blob, the loaded candidate count — never on the intention.

**L14 — Verify the committed blob, not the commit message.** Commit messages got ahead of their
contents three times, once because the command that would have written the file was blocked by
a shell guard and the commit ran anyway after an assertion failed.

**L15 — A stale rehearsal file can masquerade as a result.** rsync here has no `--delete`, so a
rehearsal row with `context_length` 4096 sat in the real run's pull directory under a name the
killed stage never wrote, and reads as a **0.18x "128k speedup"**. Pull directories are now
cleared per session and separated by `SESSION_TAG`. The guard for it is a **per-stage check
that every row's context and `max_new_tokens` agree with the config**, not a file-exists check.

**L16 — Practical traps.** A bash script must not be edited while it runs (offset reads); the
pod keeps its own staged copy of the remote script. `conda run -n rasd python - <<'PY'`
silently does nothing with a stdin heredoc — write the script to a file. The **base** miniconda
`python` on this Mac has a broken torch (`libtorch_cpu.dylib`); use the `rasd` env. The HF token
goes to the pod's `~/.hf_token` (mode 600) over stdin and is never in `argv` — passing it
inline put it in `ps` on both machines.
