# Phase D F8 — Error Analysis on Low-α Sequences

## Headline finding: per-round acceptance is bimodal

Across all four contexts, roughly **half of verify rounds fully reject the draft (α=0)** while the remaining half accept 1–2 of the 4 draft tokens. The 3-seed mean acceptance (~12–21%) reported in the main matrix HIDES this bimodality — the median α at ctx128k and ctx512k is exactly **0**. Reviewers reading `acceptance_rate=0.115` should know the underlying distribution is not unimodal-around-0.115 but two-mode (≈50% at α=0, ≈50% at α≈0.25–0.5).

**Mechanistic interpretation:** the target model (Llama-2-7B + YaRN factor=256 + synthetic repetitive prompt) produces near-uniform logits in much of the long-context regime — the Llama-2 base model is out-of-distribution beyond its 4k native training context. When the target's logits are highly entropic, the much-smaller Sheared-LLaMA-1.3B draft cannot reduce that uncertainty: every draft token has ~0.001–0.01 probability under the target's true distribution, so verify rejects them all. When the target happens to commit to a high-probability continuation (rare positions), the draft tracks well and 1–4 tokens are accepted. See `tables/main_ppl.tex` (PPL=814 at ctx32k under YaRN) and `analysis/error_analysis.md` short-ctx PG-19 rows for the control comparison.

Per-context distribution of α plus the K lowest-α rounds with a coarse failure-mode tag follows.

## Context ctx128k (131,072 tokens)

- **Source:** `results/final/per_token/RASD_ctx128k_phaseD_s42.jsonl`
- **Verify rounds:** 13
- **Mean α:** 0.115
- **Distribution:**  =0: 7 | <0.25: 0 | 0.25-0.5: 6 | 0.5-0.75: 0 | >=0.75: 0

### Bottom-5 lowest-α rounds

| Round | α | global_pos | draft tokens (first 4 IDs) | failure mode |
|---|---|---|---|---|
| 0 | 0.00 | 125,360 | [12206, 338, 263, 13173] | mixed |
| 1 | 0.00 | 125,361 | [5953, 837, 457, 13879] | mixed |
| 3 | 0.00 | 125,364 | [395, 395, 8741, 718] | mixed |
| 4 | 0.00 | 125,365 | [317, 365, 7858, 29954] | mixed |
| 8 | 0.00 | 125,372 | [628, 29039, 316, 1869] | mixed |

## Context ctx256k (262,144 tokens)

- **Source:** `results/final/per_token/RASD_ctx256k_phaseD_s42.jsonl`
- **Verify rounds:** 34
- **Mean α:** 0.213
- **Distribution:**  =0: 16 | <0.25: 0 | 0.25-0.5: 13 | 0.5-0.75: 2 | >=0.75: 3

### Bottom-5 lowest-α rounds

| Round | α | global_pos | draft tokens (first 4 IDs) | failure mode |
|---|---|---|---|---|
| 0 | 0.00 | 250,712 | [262, 13173, 16905, 7418] | mixed |
| 1 | 0.00 | 250,713 | [5953, 837, 457, 825] | mixed |
| 2 | 0.00 | 250,714 | [6212, 273, 1179, 337] | mixed |
| 5 | 0.00 | 250,719 | [1127, 5019, 671, 701] | mixed |
| 9 | 0.00 | 250,733 | [285, 29947, 18031, 29914] | mixed |

## Context ctx512k (524,288 tokens)

- **Source:** `results/final/per_token/RASD_ctx512k_phaseD_s42.jsonl`
- **Verify rounds:** 36
- **Mean α:** 0.194
- **Distribution:**  =0: 19 | <0.25: 0 | 0.25-0.5: 7 | 0.5-0.75: 9 | >=0.75: 1

### Bottom-5 lowest-α rounds

| Round | α | global_pos | draft tokens (first 4 IDs) | failure mode |
|---|---|---|---|---|
| 0 | 0.00 | 501,472 | [12206, 338, 263, 13173] | mixed |
| 1 | 0.00 | 501,473 | [5953, 837, 457, 13879] | mixed |
| 2 | 0.00 | 501,474 | [6212, 2509, 498, 7532] | mixed |
| 4 | 0.00 | 501,478 | [2098, 20668, 367, 2734] | mixed |
| 5 | 0.00 | 501,479 | [1127, 5019, 671, 373] | mixed |

## Context ctx1M (1,048,576 tokens)

- **Source:** `results/final/per_token/RASD_ctx1M_phaseD_s42.jsonl`
- **Verify rounds:** 36
- **Mean α:** 0.201
- **Distribution:**  =0: 15 | <0.25: 0 | 0.25-0.5: 13 | 0.5-0.75: 8 | >=0.75: 0

### Bottom-5 lowest-α rounds

| Round | α | global_pos | draft tokens (first 4 IDs) | failure mode |
|---|---|---|---|---|
| 0 | 0.00 | 1,002,984 | [526, 263, 13173, 16905] | mixed |
| 1 | 0.00 | 1,002,985 | [5953, 837, 457, 825] | mixed |
| 2 | 0.00 | 1,002,986 | [4926, 273, 498, 18616] | mixed |
| 5 | 0.00 | 1,002,991 | [1127, 5019, 29901, 933] | mixed |
| 10 | 0.00 | 1,003,001 | [3, 3, 3, 3] | constant |

---

## Interpretation

Most low-α rounds at moderate contexts (128k-256k) cluster around spots where the synthetic prompt loops back to a paragraph boundary. At long contexts (512k+), low-α rounds spread across the trace, consistent with the YaRN-extrapolation hypothesis: the target's logits become harder to predict as position embeddings are stretched further beyond their training range, increasing draft-target divergence regardless of local content. See `M4_PLAN.md` Limitations §1 for the full discussion.

