# Check 3 — Runtime sampling-config consistency

**Date:** 2026-04-16
**Script:** [scripts/check3_config_consistency.py](../../scripts/check3_config_consistency.py)
**Env:** laptop, torch 2.11 CPU
**Verdict:** **PASS** — no config drift between draft and target sampling paths.

## Claim under test

Every call site inside `generate()` that consumes a sampling knob
(`temperature`, `top_p`, `top_k`) reads from the *same* `RASDConfig`
instance, so the draft and target cannot see divergent values.

## Why config drift was plausible

The textbook failure mode for this bug class is Hugging Face's
`GenerationConfig`: a model may cache its own `generation_config`
(temperature baked in at load time) which silently overrides the
user-supplied value unless every call path threads its own config in
explicitly. RASD sidesteps this by calling `_sample()` and
`_acceptance_mask()` directly on logits rather than going through
`model.generate()`.

## Evidence

### 3.1 Static AST audit

All 4 call sites in [src/models/rasd_inference.py](../../src/models/rasd_inference.py)
reference `cfg.temperature` / `cfg.top_p` explicitly:

```
L509  _sample(next_token_logit, cfg.temperature, cfg.top_p)
L544  _sample(d_logit, cfg.temperature, cfg.top_p)
L628  _acceptance_mask(draft_seq, target_logits_v, draft_logits, cfg.temperature)
L662  _sample(bonus_logit, cfg.temperature, cfg.top_p)
```

Plus two direct reads inside the residual-resample branch (L664-665):

```python
t_probs_row = F.softmax(target_logits_v[:, n_acc, :] / cfg.temperature, dim=-1)
d_probs_row = F.softmax(draft_logits[:, n_acc, :]    / cfg.temperature, dim=-1)
```

Same `cfg.temperature`, applied identically on both sides of the ratio.

There is no `top_k` knob in `RASDConfig` — not supported. `_sample()` does
not accept it either.

### 3.2 Runtime trace

Monkey-patched `_sample` and `_acceptance_mask` with recording wrappers,
then exercised every call site the way `generate()` does with a config of
`(temperature=0.7, top_p=0.9, spec_steps=4)`. Recorded 7 calls, every one
with `temperature=0.7, top_p=0.9`:

```
_sample              temperature=0.7, top_p=0.9
_sample              temperature=0.7, top_p=0.9
_sample              temperature=0.7, top_p=0.9
_sample              temperature=0.7, top_p=0.9
_sample              temperature=0.7, top_p=0.9
_acceptance_mask     temperature=0.7
_sample              temperature=0.7, top_p=0.9
```

Log: [check3_config_consistency.log](check3_config_consistency.log).

## Interpretation

Structural + runtime evidence both confirm: draft and target sampling
paths share a single immutable `RASDConfig` and no `GenerationConfig` is
ever consulted. The low α observed in M3 cannot be attributed to config
drift. Combined with Check 1 (tokenizer equality) and Check 2 (verify
math), the only remaining hypothesis for the *pre-fix* low α is the
verify-loop bug — which is now patched.

## Next

Proceed to Check 4 (fp16 vs NF4 mini-ablation on pod) with the patched
code. Check 4 is now *not* a "bug or quantization?" question — the bug is
fixed. Check 4 becomes a standalone characterization: *how much α does
4-bit NF4 quantization alone cost, with correct speculative decoding?*
