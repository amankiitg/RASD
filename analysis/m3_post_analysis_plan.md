# M3 Post-Analysis: Acceptance Rate Diagnostic Study

**Context for the agent:** This is a post-hoc diagnostic study for RASD Milestone 3.
The M3 ablation observed acceptance rates (α) of 0.06–0.11 across all cells, which is
3–5× lower than published benchmarks for Llama-2-7B with TinyLlama or Sheared-LLaMA
drafts (expected α ≈ 0.25–0.45). This study determines whether the low α is explained
by the operating regime (4-bit NF4 + ring attention + 8k context) or by a bug.

**Goal:** Produce a short technical note (1–2 pages) that either (a) confirms the low α
is regime-driven and correctly characterized, or (b) identifies a bug to fix before the
next milestone.

**Deliverable:** `analysis/m3_post_analysis.md` with results, plots, and a
one-paragraph conclusion suitable for inclusion in the M3 writeup as a
"validation of operating regime" subsection. All intermediate artifacts
(logs, plots, configs) live under `results/quant_mini/`. The RunPod entry
point is `scripts/run_quant_ablation.py`.

---

## Study design

Four diagnostic checks, ordered from cheapest to most expensive. Stop early if any
check reveals a clear bug — fix the bug, then rerun M3.

### Check 1 — Tokenizer and vocab equality (LOCAL, MacBook, ~1 min)

**Claim under test:** Draft and target use identical tokenizers and vocabularies.

**Why this matters:** Even a single different BPE merge silently craters α on any
token that straddles the mismatch. TinyLlama and Sheared-LLaMA both *claim* to use
the Llama-2 tokenizer, but verify.

**Procedure:**
1. Load all three tokenizers: `meta-llama/Llama-2-7b-hf`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`,
   `princeton-nlp/Sheared-LLaMA-1.3B`.
2. Assert vocab size equality (all should be 32000).
3. Assert full vocab dict equality: `tokenizer_a.get_vocab() == tokenizer_b.get_vocab()`.
4. Tokenize 50 diverse probe strings (code, numbers, URLs, multilingual, emoji, whitespace
   edge cases) with each tokenizer; assert token ID sequences are identical.
5. Assert `model.config.vocab_size` matches across all three model configs.

**Pass criterion:** All three equal on all checks. **Fail criterion:** any mismatch →
log the divergent tokens/IDs and stop; this is the bug.

**Cost:** free, ~1 minute on laptop CPU.

### Check 2 — Sampling and verification code audit (LOCAL, static review)

**Claim under test:** The speculative verification code implements the correct
acceptance probability `min(1, p_target(t)/p_draft(t))` with matched temperature
and identical logits post-processing between draft and target.

**Why this matters:** Subtle bugs here are silent killers. Common failure modes:
- Temperature applied to draft but not target (or vice versa)
- Top-k or top-p filtering applied inconsistently
- Comparing `p_target` (post-softmax) against `logit_draft` (pre-softmax)
- Off-by-one in the resample-on-reject step
- Using stale KV in the target's verification pass

**Procedure:**
1. Locate the verification function in the repo (likely in `rasd/speculative/verify.py`
   or similar). Print the function with line numbers.
2. Check each of the following explicitly:
   - [ ] Target and draft logits are both divided by the same temperature before softmax
   - [ ] Any top-k/top-p filter is applied identically to both (or to neither)
   - [ ] The acceptance ratio uses post-softmax probabilities (both p_target and p_draft)
   - [ ] On rejection, the resample distribution is `max(0, p_target - p_draft)` normalized,
         not just `p_target`
   - [ ] The target's KV cache is correctly rewound on rejection (no carry-over from
         rejected speculative tokens)
3. Add unit tests that verify each of the above with hand-constructed mini distributions
   (vocab size 4, no model needed). Tests go in `tests/test_verification_math.py`.

**Pass criterion:** All five bullets verified by code inspection AND all unit tests pass.
**Fail criterion:** Any bullet fails → this is likely your bug. Fix, then retest.

**Cost:** free, ~30 minutes of careful code review + test writing.

### Check 3 — Temperature consistency trace (LOCAL, CPU with mock tensors)

**Claim under test:** The *actual runtime values* of temperature, top-k, top-p
reaching the draft's sample() and the target's verify() are identical on every step.

**Why this matters:** Check 2 verifies the code is correct; Check 3 verifies the
config is correctly propagated at runtime. Config drift is common when multiple
`GenerationConfig` objects exist.

**Procedure:**
1. Add a debug hook that logs `{temperature, top_k, top_p, do_sample}` from both the
   draft and target sampling paths on every speculative round.
2. Run a short generation (100 tokens) on laptop CPU with tiny stub models
   (`HuggingFaceTB/SmolLM-135M` as both draft and target — just for pipe-through testing).
3. Assert that draft's config == target's config on every round. Save the log.
4. Delete the debug hook before committing (or gate behind `RASD_DEBUG_SAMPLING=1`).

**Pass criterion:** Configs match on every round. **Fail criterion:** any divergence
→ fix the propagation, then retest.

**Cost:** free, ~15 minutes on laptop.

### Check 4 — fp16 vs 4-bit α comparison (RUNPOD, single A100, ~45 min)

**Claim under test:** The observed α ≈ 0.1 is primarily driven by 4-bit NF4
quantization of both models, not by a system bug.

**Why this matters:** This is the decisive diagnostic. If α jumps to 0.25+ at fp16,
the low α is regime-driven and M3's conclusions stand (with proper scoping). If α
stays near 0.1 at fp16, there is a deeper bug that the first three checks missed.

**Procedure:**

Run a **reduced-scope mini-ablation** — this is NOT a full M3 rerun. Purpose is
maximum signal per GPU-hour.

Configuration:
- **Hardware:** 1× A100 80GB (single GPU — disable ring attention for this test;
  we are isolating the quantization variable, not testing distributed correctness)
- **Target:** Llama-2-7B
- **Draft:** Sheared-LLaMA-1.3B (the M3 default)
- **Context length:** 2048 (shorter than M3's 8k — we want α for the pair, not
  the long-context effect)
- **k:** 4 (M3 default)
- **Seeds:** 2 (we are looking for a large effect; tight CIs not needed)
- **Prompts:** Use the same 10-prompt suite as M3 for direct comparison

Cells (4 total runs):
| Cell | Target quant | Draft quant | Expected α |
|------|-------------|-------------|-----------|
| C1   | fp16        | fp16        | 0.30–0.45 (published baseline) |
| C2   | fp16        | 4-bit NF4   | 0.20–0.35 (draft quant cost only) |
| C3   | 4-bit NF4   | fp16        | 0.20–0.35 (target quant cost only) |
| C4   | 4-bit NF4   | 4-bit NF4   | 0.10–0.20 (M3 regime, shorter ctx) |

Run each cell with 2 seeds → 8 runs total. At ~5 min/run on single A100 ≈ 40 min wall
clock + setup. Budget: **1.5 hours of A100 time ≈ $2**.

Memory budget per cell:
- fp16 Llama-2-7B: ~13 GB weights + ~2 GB KV @ 2k + overhead ≈ 20 GB → fits
- fp16 Sheared-1.3B: ~2.5 GB weights → fits alongside
- Total peak: ~28 GB on an 80GB A100 → comfortable

**Decision tree on results:**

```
                   α at C1 (fp16/fp16)
                  /                    \
              ≥ 0.30                  < 0.20
                │                        │
         Regime is correct.         Bug is NOT quantization.
         Compare C4 to M3's α:      Revisit Check 2 — the
           - If close: regime        verification math is
             story validated.        likely wrong. Do not
           - If C4 >> M3:            proceed to next
             ring-attn or 8k-ctx     milestone until resolved.
             effect is bigger
             than expected — run
             one 8k cell to
             localize.
```

**Pass criterion:** C1 ≥ 0.30 AND C4 within 0.05 of M3's observed α at matched conditions.
Both must hold → the M3 regime is correctly characterized and low α is explained.

**Fail criterion:** C1 < 0.20 → bug hunt. File an issue and stop the milestone 4
kickoff until resolved.

---

## Execution plan

The agent should proceed as follows:

1. **Start local.** Run Checks 1, 2, 3 in order on the MacBook. Do not touch RunPod yet.
2. **Gate on local results.** If any local check fails, fix the bug and loop back.
   Only proceed to Check 4 once Checks 1–3 are green.
3. **Draft the RunPod script** for Check 4 locally. Include:
   - A `scripts/run_quant_ablation.py` entrypoint that takes
     `--target-quant {fp16,nf4}` and `--draft-quant {fp16,nf4}` as args
   - Uses `bitsandbytes` for NF4 quant (same as M3)
   - Logs per-round α, tps, accepted-token counts to JSON
   - Can run a single cell end-to-end in < 10 minutes
4. **Dry-run the script** on CPU with tiny stub models to catch shape/logic bugs
   before burning RunPod time.
5. **Provision RunPod.** A single A100 80GB PCIe or SXM instance. PCIe is fine — we
   are not using NVLink. ~$1.10/hr on RunPod community.
6. **Run the 4 cells × 2 seeds = 8 runs.** Collect JSON logs.
7. **Analyze.** Compute per-cell mean α and 95% bootstrap CIs. Plot as a 2×2 grid
   (same style as the M3 bar chart).
8. **Write `analysis/m3_post_analysis.md`** with:
   - One-paragraph summary of findings
   - Table of per-cell α
   - Bar chart
   - Decision-tree outcome (regime validated / bug found)
   - If validated: a 3-sentence "scope and limitations" paragraph ready to paste into
     the M3 writeup

## Output format requirements

For each check, the agent must produce:
- A clear **PASS / FAIL / INCONCLUSIVE** verdict
- The raw evidence (log snippet, assertion output, or plot)
- One-sentence interpretation
- Next action (proceed / fix bug X / escalate)

## Constraints

- **Do not** modify any production RASD code as part of this study. Diagnostic hooks
  go behind env-var gates; unit tests go in `tests/`.
- **Do not** re-run the full M3 ablation. This is a targeted post-hoc study.
- **Do** save all intermediate artifacts (logs, plots, configs) in
  `results/quant_mini/` for reproducibility.
- **Budget cap:** $5 of RunPod time. If Check 4 exceeds this, stop and escalate.