# Check 1 — Tokenizer + vocab equality

**Date:** 2026-04-16
**Script:** [scripts/check1_tokenizer_equality.py](../../scripts/check1_tokenizer_equality.py)
**Env:** laptop, `transformers==5.5.4`, CPU-only
**Verdict:** **PASS** — draft/target tokenizers are byte-identical. No tokenizer-level α loss.

## Sub-check results (all PASS)

| # | Claim | Result |
|---|---|---|
| 1.1 | `vocab_size == 32000` on all three | PASS — Llama-2-7b-hf, TinyLlama_v1.1, Sheared-LLaMA-1.3B all report 32000 |
| 1.2 | `get_vocab()` dict equality pairwise vs Llama-2-7b-hf | PASS — TinyLlama and Sheared both byte-equal |
| 1.3 | Token-ID identity on 50 probe strings | PASS — 100/100 pairings (50 probes × 2 non-ref tokenizers) match |

## Probe coverage

50 probes across: plain English, Python/SQL/JS code, URLs + filesystem paths,
numbers (int/float/hex/commas/ISO-8601/IPv4), unicode (Greek/Cyrillic/Hebrew/Arabic/
CJK), emoji + flags, whitespace edges (leading/trailing/mixed CRLF), BPE-straddling
words, markdown structure, long repeated content, and empty/whitespace-only strings.

## Raw log

```
========================================================================
Check 1: tokenizer + vocab equality
========================================================================
  loaded  meta-llama/Llama-2-7b-hf
  loaded  TinyLlama/TinyLlama_v1.1
  loaded  princeton-nlp/Sheared-LLaMA-1.3B

1.1  vocab_size == 32000 across all three
  [PASS]  meta-llama/Llama-2-7b-hf: vocab_size = 32000
  [PASS]  TinyLlama/TinyLlama_v1.1: vocab_size = 32000
  [PASS]  princeton-nlp/Sheared-LLaMA-1.3B: vocab_size = 32000

1.2  get_vocab() dict equality vs reference meta-llama/Llama-2-7b-hf
  [PASS]  TinyLlama/TinyLlama_v1.1 vs meta-llama/Llama-2-7b-hf: equal
  [PASS]  princeton-nlp/Sheared-LLaMA-1.3B vs meta-llama/Llama-2-7b-hf: equal

1.3  Token-ID identity across 50 probe strings
  [PASS]  Probe-string ID identity: 100 pairings

  PASS — draft/target tokenizers are byte-identical. No tokenizer-level α loss.
```

## Interpretation

All three models share the identical Llama-2 SentencePiece tokenizer (vocab=32000).
Token-ID identity on a diverse 50-probe suite rules out a BPE-merge mismatch as
a cause of the observed low α. The bug lies elsewhere — see
[check2_audit.md](check2_audit.md).
