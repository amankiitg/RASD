#!/usr/bin/env python3
"""MLSys Phase 0.2 — tokenizer identity for the Llama-3 native arms.

The native-vs-YaRN experiment (arms 2 and 3) runs Llama-3.2-1B as the
draft for Llama-3.1-8B as the target. Cross-family speculation is only
sound if the two share the *identical* tokenizer, so this checks that
before any GPU time is spent:

  1. vocab_size equality (Llama-3.x is 128256, NOT Llama-2's 32000)
  2. get_vocab() dict equality (token -> id)
  3. token-ID identity on the full probe corpus, reused from
     scripts/check1_tokenizer_equality.py so the two checks cover the
     same strings (code, URLs, multilingual, emoji, BPE-straddling cases)

Exit codes:
  0 = identical (native arms are safe to run)
  1 = DIVERGENT (do NOT run arms 2/3; report and skip the native arms,
      and make sure arxiv/REVIEW_ROADMAP.md reflects that)
  2 = inconclusive (auth/network — Llama-3.x are gated repos)

Both repos are gated: export HF_TOKEN (or place it in runpod_creds.md,
which this script reads the same way check1 does).

Usage:
    python scripts/mlsys_check_tokenizer_llama3.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reuse check1's credential bootstrap and probe corpus verbatim so the
# two checks cannot drift apart.
if "HF_TOKEN" not in os.environ:
    creds = REPO / "runpod_creds.md"
    if creds.exists():
        m = re.search(r"HF_TOKEN\s*=\s*(\S+)", creds.read_text())
        if m:
            os.environ["HF_TOKEN"] = m.group(1)
            os.environ["HUGGINGFACE_HUB_TOKEN"] = m.group(1)

from scripts.check1_tokenizer_equality import PROBES  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

MODELS = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
]

# Llama-3.x SentencePiece/BPE vocab. Recorded so a mismatch is reported
# as "different tokenizer family", not merely "different size".
EXPECTED_VOCAB = 128256


def header(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def sub_result(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}]  {name}"
          + (f"  — {detail}" if detail else ""))
    return passed


def main() -> int:
    header("Phase 0.2 — Llama-3.1-8B vs Llama-3.2-1B tokenizer identity")

    tokenizers = {}
    for name in MODELS:
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(name)
            print(f"  loaded  {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR loading {name}: {e}")
            print("  (Llama-3.x are gated — is HF_TOKEN set and accepted?)")
            return 2

    ref_name, ref = next(iter(tokenizers.items()))
    all_pass = True

    # Authoritative size is len(tokenizer) == base BPE + added special
    # tokens.  `tokenizer.vocab_size` is only the base BPE table and
    # reports 128000 for Llama-3.x; comparing THAT to 128256 is a false
    # divergence signal (it cost a full 8xA100 launch once).
    header(f"2.1  vocabulary size (len(tokenizer); Llama-3.x = {EXPECTED_VOCAB})")
    lengths = {}
    for name, tok in tokenizers.items():
        lengths[name] = len(tok)
        note = ""
        if len(tok) != EXPECTED_VOCAB:
            note = (f"vs {EXPECTED_VOCAB} expected for Llama-3.x "
                    f"(base vocab_size={tok.vocab_size})")
        sub_result(f"{name}: len(tokenizer) = {len(tok)}", True, note)
    # Verdict is driven by CROSS-MODEL agreement, not by matching a
    # constant: if the two models agree with each other, the native arms
    # are sound regardless of the constant above.
    if len(set(lengths.values())) != 1:
        all_pass &= sub_result(
            "cross-model vocabulary length", False, f"disagree: {lengths}")

    header(f"2.2  get_vocab() equality vs {ref_name}")
    ref_vocab = ref.get_vocab()
    for name, tok in tokenizers.items():
        if name == ref_name:
            continue
        v = tok.get_vocab()
        eq = v == ref_vocab
        detail = ""
        if not eq:
            only_ref = set(ref_vocab) - set(v)
            only_new = set(v) - set(ref_vocab)
            mism = [(t, ref_vocab[t], v[t]) for t in set(ref_vocab) & set(v)
                    if ref_vocab[t] != v[t]][:3]
            detail = (f"|only_ref|={len(only_ref)}, |only_new|={len(only_new)}, "
                      f"sample={mism}")
        all_pass &= sub_result(f"{name}: {'equal' if eq else 'DIFFER'}", eq, detail)

    header(f"2.3  token-ID identity across {len(PROBES)} probe strings")
    divergences = []
    for probe in PROBES:
        ref_ids = ref(probe, add_special_tokens=False)["input_ids"]
        for name, tok in tokenizers.items():
            if name == ref_name:
                continue
            ids = tok(probe, add_special_tokens=False)["input_ids"]
            if ids != ref_ids:
                divergences.append({
                    "probe": probe[:60] + ("..." if len(probe) > 60 else ""),
                    "ref_ids": ref_ids[:16], "got_ids": ids[:16],
                })
    ok = not divergences
    all_pass &= sub_result(
        f"Probe-string ID identity: {len(PROBES)} probes", ok,
        "" if ok else f"{len(divergences)} divergences (first 3 below)")
    for d in divergences[:3]:
        print(f"    probe: {d['probe']!r}")
        print(f"      ref -> {d['ref_ids']}")
        print(f"      got -> {d['got_ids']}")

    header("Phase 0.2 verdict")
    if all_pass:
        print("  PASS — Llama-3.1-8B and Llama-3.2-1B share an identical")
        print("         tokenizer. Native arms (2 and 3) are safe to run.")
        return 0
    print("  FAIL — tokenizer divergence. SKIP the native arms (2 and 3) and")
    print("         report this: cross-family speculation is unsound here.")
    print("         Run only arm 1 + everything that does not need Llama-3.")
    print("         (rc=1 => session marks SKIP_NATIVE_ARMS and CONTINUES)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
