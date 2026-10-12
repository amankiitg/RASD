# CLAIMS.md — what the paper can say, and what it must not

One page. Each claim names the exact evidence rows it rests on. Conditions are kept apart:
**A** = sampling T=0.7 seed 42 (session `cad0d8cdaf2e4994ae7a70322ef1efe3`, 2026-10-12);
**B** = greedy T=0.0 seed 42 (session `12e9c5d794254c498d741261bde0877b`, 2026-10-10/11).
A's rows live in `results/mlsys/sampling_session/`; B's in the campaign's session
workspace pull directory (`session_a_pull/`; see `results/mlsys/RESULTS_OCT.md` §2);
Phase 3's in `results/mlsys/profile/`.

---

## CAN claim

| # | claim | exact evidence |
|---|---|---|
| C1 | At **1M context**, 8x A100 80GB SXM4, NF4 KV, windowed 1B draft, T=0.7 sampling, speculative decoding gives **2.37x** over our own target-only path (0.12077 vs 0.05103 tok/s). | `SAMP_1M_SPEC_pg19_train_{0,1,915,2204,2768}_s42` (5 ok rows, mean tok/s from `tokens_generated/time_sec`) ÷ `SAMP_1M_TARGET_pg19_train_{0,1}_s42` (2 rows) |
| C2 | That result is **not a greedy artifact**: **0 of 26** sampled rows are degenerate under the campaign's absolute repetition rule, against **9 of 15** rows under greedy. | `repetition_*.csv` in `results/mlsys/sampling_session/` (one per stage); longest verbatim repeat span 8 tokens vs a 128 ceiling, tokens-in-repeat 0.00 vs a 0.60 ceiling |
| C3 | Under sampling, acceptance is **stable-to-rising with context** on the aligned pools: 0.4349 (128k, n=3) → 0.4892 (512k, n=3) → **0.5246** (1M, n=5), and the speedup is **roughly flat** (2.38 / 2.59 / 2.37x) across a factor of 8 in context. | `SAMP_ALIGNED_128K_SPEC_*`, `SAMP_ALIGNED_512K_SPEC_*`, `SAMP_1M_SPEC_*` and their `*_TARGET` partners |
| C4 | The native-1M target **stays coherent at 1M**: within ~10% of its own 128k perplexity on the same tokens (1.805 vs 1.649 ppl on the one exactly aligned book; ≤1.1% on two books sharing 91-95% of their scored tokens). | `QUALITY_FIX_1M_*` vs `QUALITY_FIX_128K_*` (ppl_tokens 1024) + `sampling_session/alignment.csv` |
| C5 | The llama3 rope intervention is **nearly free at 128k** on a model that ships the dict: `ppl_ratio` 1.0005 (factor 16) and 1.0156 (factor 32) against native, `gate_pass=True` on all three arms, window unchanged at 131072. | `session_a_rope_fix_gate.csv` (ROFIX_F16 / ROFIX_F32 / ROFIX_NATIVE) |
| C6 | The target-only decode runs **42-59x above a derived per-token traffic floor** (59/232/461 ms), and the measured time is nearly **linear in the number of ring chunk transfers** (0.58-0.86 ms each over an 8x range of context). | `results/mlsys/BANDWIDTH_FLOOR.md` (14 assumptions tabulated) + the measured rows it cites |
| C7 | The llama3 rope intervention at 128k **costs speculation acceptance**: **0.7092 native → 0.5002 (factor 16) / 0.4990 (factor 32)**, a 0.21 absolute (29% relative) fall that holds on all three books individually. It is attributable to the intervention because nothing else differs between the arms — same target and revision (`meta-llama/Llama-3.1-8B` `d04e592b…`), same draft and revision (`meta-llama/Llama-3.2-1B` `4e20de36…`, `draft_window_cap` 4096), same books, seed 42, temperature 0.7, `spec_steps` 4, 128 tokens; **only `rope_factor` changes**. | `session_a_ROPE_SAMP_{NATIVE,F16,F32}_SPEC.csv` (n=3 each, same 3 books) + `session_a_rope_fix_gate.csv` |
| C8 | **The gap is not solely a ring-communication phenomenon.** With the ring absent entirely (`--nproc 1`, `comm_us = 0.0` exactly), the engine is still **28x above the floor** (1.658 s per decode token at 128k), and **87%** of the measured row is recorded local op time. Ring attention multiplies per-token decode by **~2.1x** (3.526 vs 1.658 s/token). | `results/mlsys/profile/profile_a_nproc1.csv` + both sidecars in `profile/profiler/`; the 8-rank row `SANITY_targetonly_pg19_train_0_s42` |
| C9 | Under greedy (condition B, kept separate): 1M clean n=1, acceptance 0.2132, 0.08411 vs 0.05236 tok/s = **1.61x**; 512k 1.84x. | B's `SUNDAY_HEADLINE_1M_SPEC/TARGET_*`, `SUNDAY_SCALING_512K_*` |

## Must NOT claim

| # | forbidden claim | why |
|---|---|---|
| N1 | Any **1M acceptance mean over documents** beyond n=5, or any distributional statement about how often a model loops. | 5 usable books, 1 seed, 1 temperature; one book (1726) was excluded as a data defect. |
| N2 | "Acceptance is flat or improves with context" as a property of the method. | Greedy (B) showed a **decline** (0.3273→0.2705→0.2132, n=1); sampling (A) shows a rise. Both are n≤5 and A's cross-rung windows share only 91-100% of their scored tokens. C3 states the measured shape, it does not generalize it. |
| N3 | "The model **benefits from** long context", or any quality *improvement* at 1M. | The one exactly-aligned book is **worse** at 1M (+9.5% ppl); the other two differ by ≤1.1% on non-identical windows. Coherence is the claim (C4), benefit is not. |
| N4 | Any statement about **production serving**, or a speedup against vLLM/any other engine. | The comparator is our own target-only path (0.05103 tok/s at 1M), which the profile shows is itself ~28x above its floor. **No vLLM baseline exists** for this model or these prompts. |
| N5 | That sampled decoding is **lossless in the bit-exact sense**. | At T>0 the engine uses `min(1, p_target/p_draft)` rejection with a residual resample; equivalence is distributional, and a sampled row cannot be compared to target-only by token identity. |
| N6 | Any **128k speedup for condition B**. | The 128k target-only arm was lost to the Mac-side SSH drop; B has no 128k denominator. |
| N7 | Any **perplexity** number from the earlier quality rows. | Those scored `ppl_tokens` = 8; the redo is 1024 (C4) and supersedes them. |
| N8 | That the **ring is the dominant cause** of the 42-59x gap, or any 2-rank attribution of time to communication. | Profile (a) shows 0% communication and a large local cost (C7); profile (b) was **skipped** — no 2-GPU A100 SXM type exists and the only 2-GPU A100 (PCIe) has no capacity. |
| N9 | A breakdown of the **local** cost into dequantization vs Python vs kernel launches. | The sidecar's three categories cannot do it, and the change needed to attribute by operator is an engine change, forbidden here because it would invalidate every timing quoted. |
| N10 | "1.658 s at 1 rank is the local share of 3.526 s at 8 ranks." | The 1-rank run holds the whole 128k KV on one rank, so it is not a like-for-like per-rank baseline. Only the weaker statement in C7 holds. |
| N11 | That all six configured 1M books ran. | `pg19_train_1726` failed twice on a 765-token continuation against a 1024-token requirement, and is reported as an exclusion. |
| N12 | That the rope acceptance drop is **explained by a perplexity cost**. | The gate's paired ppl_ratio is 1.0005 (f16) and 1.0156 (f32) — the intervention is nearly free *in likelihood* on the gate's continuation, while acceptance falls 29%. That tension is an open observation to investigate, not something to narrate away. Nor may the drop be claimed for any context other than 128k, any factor above 32, or any n above 3. |
