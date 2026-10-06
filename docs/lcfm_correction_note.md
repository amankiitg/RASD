# Correction note: YaRN anchoring in the long-context experiments

**Status:** draft for author decision. Not applied to `manuscript/`.

## Summary

The long-context (128k and above) configurations in this paper were built with a
YaRN `rope_scaling` block whose `original_max_position_embeddings` key was **not
read** by the pinned `transformers` release. The effective anchor therefore came
from `config.max_position_embeddings`, which our code sets to the *target context
length*. Every YaRN cell ran anchored on its own context length rather than on the
model's native window (4096 for Llama-2-7B), so the interpolation was applied as
if the model had been pretrained at 128k–1M tokens.

This note states the bug, its measured effect, which published results are
affected, and which transformers versions are affected and fixed.

## The bug

`transformers` 4.47.1 `_compute_yarn_parameters` (`modeling_rope_utils.py:261`)
carries the comment

```python
# TODO (joao): use the new original_max_position_embeddings from rope_scaling
```

and never reads that key for the correction band. The band boundaries — the
dimension range over which YaRN transitions from "keep the original frequency" to
"stretch by 1/factor" — are computed from `config.max_position_embeddings`. Our
model builder sets `hf_cfg.max_position_embeddings = context_length`, so the band
was placed for a model that had been trained at the long context, not for the 4k
window Llama-2 actually saw.

The consequence is that the slowest rotary channel pairs receive **more** stretch
than YaRN's design intent, because they are only partially inside a band that has
been pushed out to the long context.

### Measured effective stretch (Llama-2-7B, head_dim 128, base 10000)

Computed from the config alone by calling the real `_compute_yarn_parameters`
(no weights, no GPU). "Stretch" is the ratio of the YaRN frequency to the native
frequency for a channel pair; ideal YaRN is exactly `1/factor` for the slowest
pairs. The right-hand column is the most-overshot channel's rotation angle at the
target context, as a multiple of the largest angle that channel reaches inside the
training window — i.e. how far past the trained regime it is driven.

| factor | anchor | mean stretch (slowest 18 pairs) | max stretch | angle at 32k / training max |
|--------|--------|--------------------------------:|------------:|----------------------------:|
| 8  | context length (bug) | 0.3292 | 0.6150 | **4.92x** |
| 8  | 4096 (intended)      | 0.1250 | 0.1250 | **1.00x** |
| 32 | context length (bug) | 0.2573 | 0.5737 | **4.59x** |
| 32 | 4096 (intended)      | 0.0312 | 0.0313 | **0.25x** |

Two things to read off this table. First, the anchored-on-context cases stretch
the slowest channels by 0.3292 and 0.2573 where YaRN intends 0.1250 and 0.0312 —
2.6x and 8.2x too much. Second, with the correct anchor the slowest channels land
at exactly 1.00x of their training maximum (factor 8), which is YaRN's design
intent; with the bug they are driven 4.92x past it.

## Measured effect

Llama-2-7B, YaRN factor 8, 32k context, perplexity on a held-out PG-19
continuation, single A100 40GB, `results/anchor_ppl/anchor_ppl.csv`:

| configuration | effective anchor | PPL at 32k |
|---------------|-----------------:|-----------:|
| native, no scaling, 4k | — | 2.9746 |
| YaRN f8, anchored on context (as published) | 32768 | **826.9041** |
| YaRN f8, correctly anchored on 4096 | 4096 | **14.4526** |

The published anchoring reproduces PPL ≈ 815, the value reported in the paper.
Correct anchoring reduces it by 57x, to 14.45 — but that is still 4.9x worse than
the unscaled 4k baseline.

## Residual degradation is not explained by the anchor

Perplexity is not the whole story, and fixing the anchor does not fix the model.
Under **both** anchors the 200-token generations collapse into newline emission,
while the native 4k configuration does not:

| configuration | generated chars | blank lines | alphabetic share |
|---------------|----------------:|------------:|-----------------:|
| native 4k | 693 | 3/15 (20.0%) | 95.0% |
| YaRN f8, anchor 4096 | 264 | 176/178 (98.9%) | 95.9% |
| YaRN f8, anchor context | 302 | 146/149 (98.0%) | 93.5% |

Both YaRN settings are fluent for roughly 15–20 tokens and then emit
approximately 185 newlines and stop. So: training-free YaRN applied to a
4k-native model degrades generation at large factors *even when correctly
anchored*, and the anchoring bug is a large contributor to the perplexity damage
rather than its sole cause.

One caveat on that measurement: the generation used a 512-token prompt, so it
exercises the rope configuration but not long-range retrieval. It shows that the
configuration is damaged at short context; it does not by itself measure
long-context quality. The claim we can defend is the weaker one: the anchor
explains most of the perplexity blow-up, but not the degeneration.

## Which results stand, and which are confounded

**Stand (not rope-dependent).** The system measurements do not depend on the
quality of the model's outputs: throughput, peak and reserved memory, KV-cache
occupancy, communication/compute overlap, time-to-first-token, and the ring-
attention scaling behaviour. These are measurements of the serving stack, and the
stack did the same work regardless of which rope anchors were used. The
ring-attention communication results in particular are unaffected.

**Confounded (rope-dependent).** Anything that measures *what the target model
produced* is affected wherever a YaRN configuration was in use:

- speculative-decoding **acceptance rates** at 128k and above — a target whose
  outputs are degenerate will propose different continuations, and acceptance is
  a function of the target's distribution;
- the derived **speedups** at those contexts, since they are computed from
  acceptance;
- the **dose-response / factor ladder** (128k → 1M) wherever it changes the rope
  configuration rather than only the context length.

The distinction matters for the headline claim. The claim that acceptance
*deteriorates as context grows* is not separable, in the published runs, from the
claim that the *rope configuration* was mis-anchored at those contexts. Those two
explanations must be separated before the acceptance trend at 128k+ is asserted;
the run that separates them is the native-target configuration (Llama-3.1-8B with
its shipped `llama3` rope, untouched, inside its native window), which is unaffected
by this bug.

## Affected and fixed transformers versions

| version range | behaviour |
|---------------|-----------|
| ≤ 4.50 | `original_max_position_embeddings` is read **nowhere**; the correction band always comes from `config.max_position_embeddings`. **Affected.** |
| 4.51 – 4.55 | First releases that read the key (`4.51.0`), but they **also override the user's `factor`**, replacing it with `max_position_embeddings / original_max_position_embeddings`. Reading the key is necessary but not sufficient; an explicit `factor` is ignored. **Still affected unless the factor happens to equal that ratio.** |
| ≥ 4.56.2 | The key is read for the correction band only, and an explicitly supplied `factor` is respected. **Fixed.** |

The project remains pinned at 4.47.1 deliberately. Upgrading does not
retroactively repair the runs, and moving into 4.51–4.55 would replace one
anchoring problem with a factor-override problem. The correct-anchor behaviour was
obtained instead by threading an explicit anchor through the configuration
(`rope_anchor_base`), which is version-independent and keeps the pinned release.

## Proposed wording for the paper

> The 128k-and-above configurations in this paper were built with a YaRN
> `rope_scaling` block whose `original_max_position_embeddings` key is silently
> ignored by `transformers` ≤ 4.50, so the interpolation anchor defaulted to the
> target context length instead of the model's native 4096 window. With the
> intended anchor, Llama-2-7B perplexity at 32k falls from 826.9 to 14.45 against
> a native 4k baseline of 2.97. Our system-level measurements (throughput,
> memory, communication overlap, time-to-first-token) are unaffected; acceptance
> rates and the speedups derived from them at 128k and above are confounded with
> this configuration defect and should be re-measured with a correctly anchored
> or natively long-context target.

## Reproduction

- Stretch table: computed from config alone via `_compute_yarn_parameters`, no
  weights or GPU.
- Perplexity and generations: `scripts/anchor_ppl_probe.py`,
  `results/anchor_ppl/anchor_ppl.csv`, `results/anchor_ppl/gen_*.txt`.
- Blank-line and alphabetic shares are computed over the saved generations.
