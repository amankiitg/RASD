PROPOSED MANUSCRIPT REPLACEMENTS — NOT APPLIED (staged for separate application)
================================================================================
Finding: the loader never set `bnb_4bit_quant_type`, so bitsandbytes' DEFAULT
fp4 applied. Model WEIGHTS are therefore 4-bit FP4, not NF4. The KV cache is
genuinely NF4 (custom chunked codec, kv_quant=True). Several sentences
describe the WEIGHTS as NF4, which is factually wrong.

Two vocabulary fixes:
  * weights: "NF4-quantized weights" -> "4-bit FP4 weights (bitsandbytes)"
  * KV cache: "NF4" is CORRECT, keep as "NF4 KV cache"

--------------------------------------------------------------------------------
workshop/main.tex line 236   [WEIGHTS — needs rewording]
--------------------------------------------------------------------------------
CURRENT:
  node). Both the target model (Llama-2-7B) and a draft model
  (Sheared-LLaMA-1.3B \citep{xia2023sheared}, which shares the
  target's tokenizer) are loaded on every rank with NF4-quantized
  weights.

PROPOSED:
  node). Both the target model (Llama-2-7B) and a draft model
  (Sheared-LLaMA-1.3B \citep{xia2023sheared}, which shares the
  target's tokenizer) are loaded on every rank with 4-bit FP4 weights
  via bitsandbytes \citep{dettmers2022bitsandbytes}.

RATIONALE: this sentence is about weight precision; the KV cache (NF4) is
introduced separately in the same paragraph and is unaffected.

--------------------------------------------------------------------------------
workshop/main.tex line 241   [WEIGHTS — needs rewording]
--------------------------------------------------------------------------------
CURRENT:
  recent 4{,}096 tokens of the sequence (its KV cache covers that
  truncated window and is kept in bf16), while its weights are
  NF4-quantized like the target's. The draft's forward cost is
  therefore nearly context-independent, ...

PROPOSED:
  recent 4{,}096 tokens of the sequence (its KV cache covers that
  truncated window and is kept in bf16), while its weights are 4-bit
  FP4-quantized like the target's. The draft's forward cost is
  therefore nearly context-independent, ...

NOTE: the surrounding "KV cache ... kept in bf16" clause is correct and is
left untouched.

--------------------------------------------------------------------------------
workshop/main.tex line 626   [WEIGHTS — needs rewording, mechanism claim]
--------------------------------------------------------------------------------
CURRENT:
  \textbf{An unruled-out mechanism.} The draft and target models
  are independently NF4-quantized; divergence between their
  independently quantized logits could contribute to rejection
  rounds. The proposed isolation is a bf16 draft at 64\,k, deferred

PROPOSED:
  \textbf{An unruled-out mechanism.} The draft and target models
  are independently quantized to 4-bit FP4 weights via bitsandbytes;
  divergence between their independently quantized logits could
  contribute to rejection rounds. The isolation is a bf16 draft
  against an FP4 target at 64\,k (reported in \S\,Phase~3).

RATIONALE: this is the hypothesis the Phase-3 isolation tests. The
experiment was run and the result is that bf16-vs-FP4 acceptance is
essentially unchanged (0.0950 -> 0.0968), so this mechanism is now
RULED OUT rather than "deferred"; the wording is updated to say so.

--------------------------------------------------------------------------------
ALSO FLAGGED (not staged — judgement calls)
--------------------------------------------------------------------------------
* arxiv/main.tex 108 — keyword list: "KV cache quantization, NF4
  quantization". Ambiguous: "NF4 quantization" reads as weights. Suggest
  "KV cache quantization (NF4), 4-bit FP4 weight quantization".
* arxiv/main.tex 540/542 — "both quantized to NF4 4-bit weights via
  bitsandbytes" -> "both quantized to 4-bit FP4 weights via bitsandbytes".
  The following clause "The KV cache is also NF4-quantized" is CORRECT
  and must stay.

CONFIRMED CORRECT (KV cache, no change needed):
  workshop 69, 169, 178, 256, 261, 277, 585
  arxiv    96, 138, 165, 177, 229, 364, 366

---

## Proposed footnote: nominal vs actual synthetic prompt length

**Applies to BOTH papers** (workshop and LCFM). One sentence, placed wherever
context lengths are first quoted for synthetic-prompt cells.

Proposed text:

> Context lengths for synthetic prompts are nominal; the prompt builder
> produced ~95% of nominal (e.g. ~125k tokens for the 128k cell, and ~0.956x
> at every length from 64k to 1M), and all cells are shortened by the same
> factor, so no paired comparison is affected.

More explicit variant, if the venue wants the mechanism:

> Context lengths for synthetic prompts are nominal rather than exact: the
> builder's repetition count included the tokenizer's leading BOS, so the
> engine received ~95.6% of the nominal length (Llama-2-7B: 62660/125360/
> 250716/501472/1002984 for 64k/128k/256k/512k/1M; Llama-3.1-8B: 124803 for
> 128k). The shortfall is a near-constant factor (0.952-0.957) that each spec
> cell shares with its paired target-only baseline, so throughput ratios and
> the dose-response ladder are unaffected.

Numbers are reproducible from `results/mlsys/nominal_vs_actual_context.csv`.

**Do NOT** describe the synthetic cells as running at exactly 128k/256k/512k/1M
tokens. The ARM4 cells are exact (130944 = 131072 - 128) and need no footnote.
