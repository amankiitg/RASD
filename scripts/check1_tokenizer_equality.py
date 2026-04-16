#!/usr/bin/env python3
"""Check 1 of the M3 post-analysis: tokenizer + vocab equality.

Claim under test: Llama-2-7B, TinyLlama-v1.1, and Sheared-LLaMA-1.3B all use
the *identical* Llama-2 SentencePiece tokenizer (vocab=32000).

A single divergent BPE merge silently craters speculative decoding acceptance
on any token that straddles the mismatch, so verify exhaustively.

Sub-checks (all must PASS):
  1. vocab_size == 32000 for all three
  2. get_vocab() dict equality pairwise
  3. Token-ID identity on 50 diverse probe strings (code, URLs, multilingual,
     emoji, whitespace, numbers)
  4. (optional) model.config.vocab_size match — skipped here since we don't
     load weights locally; the tokenizer vocab is the authoritative source.

Requires HF_TOKEN env var (Llama-2 is gated). Reads from runpod_creds.md
automatically if present.

Usage:
    python scripts/check1_tokenizer_equality.py
Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (e.g., auth failed).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- Load HF_TOKEN from runpod_creds.md if not already in env ----
if "HF_TOKEN" not in os.environ:
    creds = REPO / "runpod_creds.md"
    if creds.exists():
        m = re.search(r"HF_TOKEN\s*=\s*(\S+)", creds.read_text())
        if m:
            os.environ["HF_TOKEN"] = m.group(1)
            os.environ["HUGGINGFACE_HUB_TOKEN"] = m.group(1)

from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "TinyLlama/TinyLlama_v1.1",
    "princeton-nlp/Sheared-LLaMA-1.3B",
]

PROBES = [
    # plain English
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'd've thought you'd know by now.",
    # code
    "def foo(x: int) -> int:\n    return x * 2",
    "const x = {a: 1, b: [2, 3]};",
    "```python\nimport numpy as np\nnp.zeros((4, 4))\n```",
    "select * from users where id = 42;",
    "if (x > 0) { return -x; }",
    # URLs and paths
    "https://huggingface.co/meta-llama/Llama-2-7b-hf",
    "file:///usr/local/bin/python3",
    "/Users/amankesarwani/PycharmProjects/RASD",
    # numbers and unicode
    "3.14159265358979",
    "1,234,567,890",
    "0x7fff_ffff",
    "π ≈ 3.14, √2 ≈ 1.414",
    # multilingual
    "你好,世界",
    "こんにちは世界",
    "안녕하세요 세계",
    "Привет, мир!",
    "مرحبا بالعالم",
    "שלום עולם",
    "Γειά σου κόσμε",
    # emoji + symbols
    "😀 🚀 🧠 ✅ ⚠️",
    "🇺🇸 🇯🇵 🇮🇳",
    "—…‘’“”–",
    # whitespace edge cases
    "   leading spaces",
    "trailing\t\ttabs  ",
    "mixed\n\nnewlines\rand\r\ncarriage",
    "no_separator",
    "snake_case and camelCase and kebab-case",
    # BPE-straddling cases
    "pre-tokenizer",
    "untokenizable",
    "antidisestablishmentarianism",
    "supercalifragilisticexpialidocious",
    # numbers embedded in text
    "v1.2.3-rc4+build.567",
    "2026-04-16T14:35:00Z",
    "192.168.1.1:8080",
    # special LLaMA chat tokens (should be single tokens)
    "<s>",
    "</s>",
    "<unk>",
    # text mixing scripts
    "Python3 is 80% cool — and £5 cheaper.",
    "Heisenberg's Uncertainty: Δx·Δp ≥ ℏ/2",
    # markdown / structural
    "# Heading\n- bullet\n- bullet\n",
    "| col1 | col2 |\n|------|------|\n| a    | b    |",
    # long content
    "a" * 200,
    "x y " * 100,
    # empty-ish
    " ",
    "\n",
    "\t",
    "",
]


def header(msg: str):
    print(f"\n{'='*72}\n{msg}\n{'='*72}")


def sub_result(name: str, passed: bool, detail: str = ""):
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}]  {name}" + (f"  — {detail}" if detail else ""))
    return passed


def main() -> int:
    header("Check 1: tokenizer + vocab equality")

    # ---- Load tokenizers ----
    tokenizers = {}
    for name in MODELS:
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(name)
            print(f"  loaded  {name}")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            return 2  # INCONCLUSIVE

    ref_name, ref = next(iter(tokenizers.items()))

    all_pass = True

    # ---- 1.1 vocab_size ----
    header("1.1  vocab_size == 32000 across all three")
    for name, tok in tokenizers.items():
        ok = (tok.vocab_size == 32000)
        all_pass &= sub_result(f"{name}: vocab_size = {tok.vocab_size}", ok)

    # ---- 1.2 get_vocab() equality ----
    header("1.2  get_vocab() dict equality vs reference " + ref_name)
    ref_vocab = ref.get_vocab()
    for name, tok in tokenizers.items():
        if name == ref_name:
            continue
        v = tok.get_vocab()
        eq = (v == ref_vocab)
        detail = ""
        if not eq:
            # Identify first few diverging entries for debugging
            only_ref = set(ref_vocab) - set(v)
            only_new = set(v) - set(ref_vocab)
            id_mismatches = [
                (tok_str, ref_vocab[tok_str], v[tok_str])
                for tok_str in set(ref_vocab) & set(v)
                if ref_vocab[tok_str] != v[tok_str]
            ][:5]
            detail = (f"|only_ref|={len(only_ref)}, |only_new|={len(only_new)}, "
                      f"|id_mismatches|={len(id_mismatches)} "
                      f"sample={id_mismatches[:3]}")
        all_pass &= sub_result(f"{name} vs {ref_name}: {'equal' if eq else 'DIFFER'}", eq, detail)

    # ---- 1.3 Token-ID identity on probes ----
    header(f"1.3  Token-ID identity across {len(PROBES)} probe strings")
    divergences = []
    for probe in PROBES:
        id_sets = {name: tok(probe, add_special_tokens=False)["input_ids"]
                   for name, tok in tokenizers.items()}
        ref_ids = id_sets[ref_name]
        for name, ids in id_sets.items():
            if ids != ref_ids:
                divergences.append({
                    "probe":   probe[:60] + ("..." if len(probe) > 60 else ""),
                    "model":   name,
                    "ref_ids": ref_ids[:16],
                    "got_ids": ids[:16],
                })
    ok = len(divergences) == 0
    all_pass &= sub_result(
        f"Probe-string ID identity: {len(PROBES) * (len(tokenizers) - 1)} pairings",
        ok,
        "" if ok else f"{len(divergences)} divergences (first 3 below)",
    )
    for d in divergences[:3]:
        print(f"    probe: {d['probe']!r}")
        print(f"      {ref_name:50s} -> {d['ref_ids']}")
        print(f"      {d['model']:50s} -> {d['got_ids']}")

    # ---- Verdict ----
    header("Check 1 verdict")
    if all_pass:
        print("  PASS — draft/target tokenizers are byte-identical. "
              "No tokenizer-level α loss.")
        return 0
    print("  FAIL — tokenizer divergence detected. "
          "This is a sufficient cause for the low α. Stop and fix.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
