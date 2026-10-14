# PROFILE.md — where the time actually goes, measured (Phase 3)

Instance `61d4c57d0fac41c18bcb8f6b93251f37`, `gpu_1x_a100_sxm4` (1x A100 **40 GB** SXM4),
us-east-1, **$1.99/hr**. Launched 2026-10-12T08:28:46Z, terminated 09:05:54Z, wall
2323 s = 0.65 h = **$1.28**. Detached dead-man timer (60 min) armed before any work,
confirmed by **PPID 1** (pid 31562); 0 instances after terminate, verified by the API;
timer cancelled. Evidence committed here: `results/mlsys/profile/` (the CSV, both
profiler sidecars, and the meta file). The two run logs
(`profile.log`, `decode_a_nproc1.log`) are in the same local directory but are NOT
committed — `.gitignore` excludes `*.log` repo-wide, as it does for every other stage.

**Findings only. No engine code was changed** (`src/` untouched), because any change
would invalidate every timing already measured.

---

## 1. What was run, and the one thing that differs

`bash scripts/mlsys_profile_decode.sh --nproc 1 --tag a --out results/mlsys/profile`,
config `configs/mlsys_profile_128k.yml`: target-only (`spec_steps: 0`) decode at 128k,
NF4 KV, `kv_block_size` 2048, `prefetch_depth` 1, seed 42, bf16, greedy. With
`--nproc 1` the ring is `world_size = 1`: **there is no cross-rank transfer of any kind.**

Two rows, both `status=ok`, both at `context_length` 131072:

| row | tokens | time_sec | tok/s (exact) | ms/token | ttft_ms | n_rounds | peak mem |
|---|---|---|---|---|---|---|---|
| PROF_WARM (not reported as a measurement) | 8 | 49.7945 | 0.16066 | 6224 | 38142.5 | 7 | 25,801 MB |
| **PROF_MAIN (reported)** | **24** | **76.2503** | **0.31475** | **3177** | **38121.0** | 23 | 25,799 MB |

tok/s is `tokens_generated / time_sec`, never `throughput_tps` (0.31, two significant
figures).

**The warm-up limitation is real and stated:** the runner's `--profile` wraps the whole
`generate()` call, so warm-up could only be done at ROW granularity. PROF_MAIN is the
reported row. Its first token alone costs 38.1 s of prefill (ttft), so of the 76.25 s:

```
prefill (ttft)        38.12 s   (50.0%)
decode, 23 steps      38.13 s   -> 1.658 s/token for the decode steps alone
total                 76.25 s   -> 3.177 s/token including amortised prefill
```

## 2. The profiler sidecar (verbatim, both rows)

| field | PROF_MAIN | PROF_WARM |
|---|---|---|
| `compute_us` | 53,958,745.8 (**53.96 s**) | 40,656,303.6 (40.66 s) |
| `comm_us` | **0.0** | **0.0** |
| `other_us` | 12,378,733.0 (12.38 s) | 5,460,187.2 (5.46 s) |
| `idle_us` | 59,057,530.0 (59.06 s) | 21,294,996.1 (21.29 s) |
| `total_us` | 66,337,478.8 (66.34 s) | 46,116,490.8 (46.12 s) |
| `wall_us` | 125,395,008.8 (125.40 s) | 67,411,486.9 (67.41 s) |
| `overlap_us` | 0.0 | 0.0 |
| shares of `wall_us` | compute 43.0% / comm **0.0%** / idle 47.1% / other 9.9% | 60.3% / 0.0% / 31.6% / 8.1% |

`comm_us` is **exactly 0.0** in both rows, which is what `world_size = 1` predicts: no
NCCL/ring event was recorded because none exists. This is the cleanest single fact in the
profile.

## 3. The breakdown that has to add up, and the denominator problem

**The profiler's own wall is 125.40 s while the row's measured time is 76.25 s — the
profiling itself inflates the run by 1.64x.** `torch.profiler` records every kernel, and
that is not free. So the four buckets must NOT be quoted as fractions of the measured
per-token time; they are fractions of the *profiled* wall, and the honest reading is
against both denominators:

```
of the measured 76.25 s row:
  recorded op self-time (the sidecar's total_us)   66.34 s = 87.0%
    of which compute (non-comm GPU+CPU self-time)  53.96 s = 70.8%
    of which "other" (allocator/dispatch/noise)    12.38 s = 16.2%
  communication                                     0.00 s =  0.0%
  unrecorded gaps (76.25 - 66.34)                    9.91 s = 13.0%

of the profiled 125.40 s wall:
  compute 43.0% | comm 0.0% | other 9.9% | idle 47.1%   (idle = wall - events)
```

Both framings are given because they pull in opposite directions: measured against the
run, 87% of the row is inside recorded ops and the "idle" is only 13%; measured against
the profiled wall, "idle" is 47%. The difference IS the profiler's overhead, and it lands
mostly in the CPU-side self-time that the `compute` bucket also counts. That is why
"compute 43%" must not be read as "43% of the engine's real time is compute".

## 4. The finding: the gap is not a ring-communication phenomenon

`BANDWIDTH_FLOOR.md` established that the target-only decode runs 42-59x above a
per-token traffic floor (59 ms at 128k, 461 ms at 1M) and that measured time is nearly
linear in the number of ring chunk transfers. The natural guess was that per-transfer
latency on the ring is the whole story. **Profile (a) refutes that as the whole story:**

* With the ring removed entirely (`world_size = 1`, `comm_us = 0.0`), the engine still
  spends **1.658 s per decode token** at 128k. That is **28x the 59 ms floor**, with no
  communication to blame.
* 87% of the measured row time is inside recorded ops, so the time is not hiding in
  unmeasured gaps either: it is being spent *doing something locally*, in the step.
* Nothing in the sidecar is attributable to communication. On one device the entire gap
  is local.

So the 42-59x gap has at least two large components, and the local one is proven:

**How it adds up at 128k, target-only, same config, same book, same node type**
(the 8-rank row is the sampling session's own sanity target-only row,
`SANITY_targetonly_pg19_train_0_s42`: 64 tokens, 235.0941 s, ttft 9425.3 ms):

| configuration | decode s/token (ttft removed) | vs the 59 ms floor |
|---|---|---|
| 1 rank (`nproc 1`, world_size 1), 24 tokens | 38.13/23 = **1.658 s** | **28x** |
| 8 ranks (the 8x session, 64 tokens) | (235.0941-9.4253)/64 = **3.526 s** | **60x** |

Ring attention therefore multiplies the per-token decode by about **2.1x** (3.526/1.658)
— real, and consistent with the floor analysis — but **the residual local cost is of the
same order as the entire 8-rank per-token time.** A fix aimed only at the ring would
leave most of the gap in place.

**What this measurement may NOT be used for.** The one-device run holds the whole 128k KV
on a single rank, so its *local* work per rank is larger than an 8-rank rank's (16384
tokens of local KV, not 131072). It is therefore **not** a like-for-like per-rank
local-cost baseline, and the 1.66 s must not be subtracted from the 3.53 s to give "the
ring's share". What it proves is the weaker but sufficient statement: **no communication
is required for the engine to be ~28x above its own bandwidth floor**, so the gap cannot
be attributed to bytes or to per-transfer latency alone. Splitting the local 1.66 s into
dequantization vs Python vs kernel launches would need engine changes (forbidden here) or
a profiler mode that attributes by operator; the sidecar's three categories cannot do it.

## 5. Profile (b) — SKIPPED, for two independent reasons

**(1) There is no 2-GPU A100 SXM (NVLink) instance to run it on.** Queried from the API
at 08:26Z, 09:11Z and 09:15Z (still inside the plan's two-hour window):

| type | description | $/hr | capacity |
|---|---|---|---|
| `gpu_2x_a100` | 2x A100 (40 GB **PCIe**) | 3.98 | **none, every region** |
| `gpu_1x_a100` | 1x A100 (40 GB PCIe) | 1.99 | none |
| `gpu_1x_a100_sxm4` | 1x A100 (40 GB SXM4) | 1.99 | us-east-1, us-west-2 |
| `gpu_8x_a100` | 8x A100 (40 GB SXM4) | 15.92 | us-west-2 |
| `gpu_8x_a100_80gb_sxm4` | 8x A100 (80 GB SXM4) | 22.32 | us-east-1 |

There is **no `gpu_2x_a100_sxm4` type at all**, and the only 2-GPU A100 is PCIe with zero
capacity in every region. `PROFILE_PLAN.md` says explicitly that a PCIe pair "would not
represent the same interconnect and would be a worse measurement than none" — and in any
case it is not obtainable. The remaining SXM NVLink options are 8-GPU boxes at $15.92 and
$22.32/hr, i.e. $15.92 and $22.32 for a 60-minute timer, both **over profile (b)'s $10
cap**, and an 8-GPU box is not "2-GPU" for the purpose of exposing ring traffic with
`nproc 2`.

**(2) The plan's own decision rule stops at (a).** It says: run (a); "if (a) explains most
of the measured per-token time, stop there — the cost is local and inside the step. Only
if it does not, run (b) to attribute the remainder to ring communication." Profile (a)
attributes **0%** to communication and **87%** of the measured row to recorded local ops,
with the remaining 13% unrecorded gaps on a single device. The cost is local and inside
the step. There is no "remainder to attribute to ring communication" that a 2-rank run
could attribute.

So (b) is skipped on both grounds, and the second is the stronger one: it would not change
the conclusion.

## 6. What a fix would likely be, and what it would buy

Ordered by what the profile actually supports:

1. **Attack the local step first, because it is the proven term.** At 1.658 s per decode
   token with one rank, a target-only step at 128k should be tens of milliseconds. The
   candidates in `BANDWIDTH_FLOOR.md` §4 that survive the ring's removal unchanged are
   C4 (dequantize-on-demand touching far more than the attended bytes, plus a per-layer
   kernel burst) and C5/C2's per-step synchronization structure. This is where the
   largest and most certain win is, and it does not depend on the ring at all.
2. **Then the ring**, whose measured multiplier is ~2.1x at 128k: fewer, larger transfers
   (raise `kv_block_size` from 2048 to the whole local shard) and overlap transfer with
   compute (the `wait()`-per-request in `_ring_forward` forbids it by construction, and
   `prefetch_depth` is not used as a double buffer).
3. **Keeping the ring payload in NF4** (40 KiB/token instead of 128 KiB/token) is real
   but smaller than 1 and 2 in effect, and only pays if bytes — not latency and not the
   local step — turn out to dominate, which this profile says they currently do not.

**Honest bounds on these expectations.** The 2.1x ring multiplier was measured on the
40 GB SXM4 1x/8x pair at 128k, not at 512k or 1M. Item 1's expected effect is bounded
only by "1.658 s to something much smaller"; the profile cannot say how much smaller
without the change that is forbidden here.

## 7. Limitations of this profile, stated

1. **One row per configuration, one book, one seed, greedy, 24 measured tokens.** The
   decode s/token is a mean over 23 steps and is not accompanied by a per-step
   distribution; the sidecar is a single aggregate.
2. **Warm-up is at row granularity, not step granularity** (the `--profile` flag wraps
   `generate()`), so the reported row still contains a cold CUDA context's first steps.
   PROF_WARM exists to pay that cost first, which is why the WARM row's shares differ.
3. **The profiler inflates the run by 1.64x** (125.40 s profiled vs 76.25 s measured).
   Every share in §2 is a share of the profiled wall; §3 gives the measured-time framing
   too, and the two must be read together.
4. **`compute_us` is `self_cpu_time + self_cuda_time`** (`src/analysis/profiler.py:128`),
   so "compute" is ops that are not classified as comm, with CPU dispatch time added in.
   It is not "GPU math time".
5. **The 1-GPU vs 8-GPU comparison is not a controlled experiment**: the 8-rank row comes
   from a different config file (`mlsys_sanity_sampling.yml`) at the same 128k / NF4 /
   `kv_block_size` 2048 / `prefetch_depth` 1 / bf16 settings, with T=0.7 and 64 tokens
   where the profile row is greedy with 24. Per-token decode cost is insensitive to
   temperature, but the token counts and prompt windows differ.
6. **No 2-rank measurement exists**, so the ring's contribution is inferred from the
   1-rank/8-rank pair rather than measured at 2 ranks.
