# Check 2 — Sampling/verification code audit

**Date:** 2026-04-16
**Target:** [src/models/rasd_inference.py](../../src/models/rasd_inference.py)
**Verdict:** **FAIL (3 of 5 checklist items + 1 extra)**
**Action:** Stop the diagnostic sequence. Fix the verify loop before Check 3 or 4.

## Checklist

| # | Criterion | Result | Location |
|---|---|---|---|
| 1 | Target and draft logits divided by the same temperature before softmax | PASS | [rasd_inference.py:268-269](../../src/models/rasd_inference.py#L268-L269) |
| 2 | Any top-k/top-p filter applied identically to both (or neither) | PASS | `_acceptance_mask` applies neither; sampler applies top_p only at commit |
| 3 | Acceptance ratio uses post-softmax probabilities | PASS | [rasd_inference.py:271-273](../../src/models/rasd_inference.py#L271-L273) |
| 4 | On rejection, resample from `max(0, p_target − p_draft)` normalized | **FAIL** | [rasd_inference.py:627-628](../../src/models/rasd_inference.py#L627-L628) |
| 5 | Target's KV rewound to committed length on rejection | **FAIL** | [rasd_inference.py:600](../../src/models/rasd_inference.py#L600) |
| 6 | *(extra)* Target verify-loop conditioned on draft tokens, not target's argmax | **FAIL** | [rasd_inference.py:598](../../src/models/rasd_inference.py#L598) |

## Detail

### Finding #4 — plain `p_target` on rejection

```python
bonus_logit = target_logits_v[:, n_acc, :]
cur_token   = _sample(bonus_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)
```

Leviathan et al. require the **residual** distribution
`normalize(max(0, p_target[n_acc] - p_draft[n_acc]))` so that the combined
draft-reject + resample path yields samples from `p_target` exactly. The
current code samples directly from `p_target`, which biases the output
distribution toward tokens where `p_draft` was also high — silently lowering
effective acceptance because the next draft round is seeded from a point
`p_target` alone prefers, rather than a point `p_target` prefers *given
`p_draft` got it wrong*.

### Finding #5 — KV never truncated

```python
past_kv = t_past_kv   # t_past_kv holds k+1 verify steps
```

The target's `past_key_values` is assigned after `spec_steps + 1`
autoregressive forward calls, regardless of `n_accepted`. Next round reads
past_kv as if all k+1 tokens had been committed, so the bonus `cur_token`
arrives at the wrong positional offset and every subsequent verify step
attends to rejected-token KV entries.

### Finding #6 — target runs its own continuation

```python
for _ in range(cfg.spec_steps + 1):
    t_out = self.target_model(t_input, past_key_values=t_past_kv, use_cache=True)
    ...
    t_input = t_logit.argmax(dim=-1, keepdim=True)   # <-- wrong
```

The target is fed its **own greedy argmax** as the next input — not the
draft's proposed token. That makes `target_logits_v[:, i]` the distribution
`p_target(· | cur_token, target_argmax[0], …, target_argmax[i-1])`, whereas
speculative decoding requires `p_target(· | cur_token, draft_seq[0], …,
draft_seq[i-1])`. Beyond the first verify position, the target is scoring
the draft's token against a distribution the draft never saw. Acceptance
collapses toward chance on multi-token lookaheads — a plausible direct
cause of the observed α ≈ 0.06–0.11.

## Test spec

[tests/test_verification_math.py](../../tests/test_verification_math.py) —
6 tests, all passing against a reference `speculative_verify_round_ref()`
that packs `[cur_token, draft_seq]` into a single forward, uses residual
resample on rejection, and truncates KV to `prior + n_accepted + 1`. The
test file **is the spec** the production fix must satisfy.

```
tests/test_verification_math.py::test_target_conditioned_on_draft_tokens PASSED
tests/test_verification_math.py::test_kv_truncated_to_commit_length PASSED
tests/test_verification_math.py::test_kv_length_on_full_acceptance PASSED
tests/test_verification_math.py::test_residual_resample_on_rejection PASSED
tests/test_verification_math.py::test_bonus_on_full_acceptance_is_plain_target PASSED
tests/test_verification_math.py::test_acceptance_ratio_math PASSED
```

## Recommendation

This is almost certainly *the* M3 α bug. Check 3 (runtime config trace) and
Check 4 (fp16-vs-NF4 pod run) would both reproduce α ≈ 0.1 regardless of
quantization because the defect is upstream of both. **Fix the verify loop
first, rerun a single-seed M3 smoke, and only re-open the quantization
question if α still sits below 0.25.**
