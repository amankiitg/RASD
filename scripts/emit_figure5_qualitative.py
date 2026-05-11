#!/usr/bin/env python3
"""Phase D F5: side-by-side qualitative comparison of RASD vs target-only
generations on the same (synthetic) prompt at each context length.

Reads:
    results/final/generated/<run_id>.txt   from --save-generated-text

Emits:
    tables/qualitative_examples.tex   LaTeX comparison table
    figures/fig5_qualitative_examples.txt   plain-text version

For each context length, pairs up the RASD and TARGET runs (same seed=42)
and shows the first N tokens of each generation side-by-side.

The synthetic prompt is repeated technical-English (build_prompt default),
so this is "given the same continuation task, what did each system
generate" — not a quality benchmark, just a qualitative consistency check.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

CTX_LABEL = {
    "ctx4k":    "4k (PG-19)",
    "ctx8k":    "8k (PG-19)",
    "ctx128k":  "128k",
    "ctx256k":  "256k",
    "ctx512k":  "512k",
    "ctx1M":    "1M",
}


def _excerpt(text: str, max_chars: int) -> str:
    """Return the LAST max_chars of `text`, after the last sentence boundary.

    The generated.txt file holds [prompt + continuation]; since both
    RASD and target share the same prompt, the qualitative difference
    is in the trailing generation. We snap the start to the last
    sentence boundary so the snippet starts mid-thought rather than
    in the middle of a word.

    The output is ASCII-safe: non-ASCII chars (e.g., Cyrillic from
    YaRN-degraded continuations) are replaced with '?' so the LaTeX
    \\input{} doesn't choke on missing font glyphs.
    """
    text = text.strip().replace("\n", " ").replace("  ", " ")
    if len(text) <= max_chars:
        out = text
    else:
        tail = text[-max_chars:]
        for boundary in (". ", "! ", "? "):
            idx = tail.find(boundary, 0, max_chars // 2)
            if idx > 0:
                tail = tail[idx + len(boundary):]
                break
        out = "[...] " + tail.lstrip()
    # ASCII-safe: replace any non-ASCII char (Cyrillic, em-dash, etc.)
    # with a single '?' so pdflatex (default fonts) doesn't error.
    out = out.encode("ascii", errors="replace").decode("ascii")
    # Also strip control characters (0x00-0x1F + 0x7F) — the model
    # sometimes generates literal NUL bytes at YaRN-extrapolated
    # contexts, which LaTeX rejects as "invalid character".
    out = "".join(c if 0x20 <= ord(c) < 0x7F else "?" for c in out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", default="results/final/generated")
    p.add_argument("--out-tex", default="tables/qualitative_examples.tex")
    p.add_argument("--out-txt", default="figures/fig5_qualitative_examples.txt")
    p.add_argument("--excerpt-chars", type=int, default=200,
                   help="Max chars to show per generation excerpt.")
    args = p.parse_args()

    gen_dir = REPO_ROOT / args.gen_dir
    if not gen_dir.exists():
        raise SystemExit(
            f"No generated text dir at {gen_dir}. "
            "Run experiments with --save-generated-text first."
        )

    # Pair RASD and TARGET runs by ctx label
    pairs: dict[str, dict[str, Path]] = {}
    ctx_pattern = r"ctx4k|ctx8k|ctx128k|ctx256k|ctx512k|ctx1M"
    for tf in sorted(gen_dir.glob("*.txt")):
        name = tf.stem
        m = re.match(rf"^(RASD|TARGET)_({ctx_pattern})(?:_pg19)?_phaseD_s\d+", name)
        if not m:
            # Also accept Phase C / older run-ids if they exist
            m = re.match(rf"^(M4|TARGET)_({ctx_pattern})_s\d+", name)
            if not m:
                continue
            kind = "RASD" if m.group(1) == "M4" else "TARGET"
        else:
            kind = m.group(1)
        ctx_label = m.group(2)
        pairs.setdefault(ctx_label, {})[kind] = tf

    if not pairs:
        raise SystemExit(
            f"No qualifying files in {gen_dir}. "
            "Filenames must match RASD_<ctx>_phaseD_s* or TARGET_<ctx>_phaseD_s*."
        )

    # --- LaTeX table ---
    tex_lines = [
        r"\begin{tabular}{p{0.12\textwidth}p{0.40\textwidth}p{0.40\textwidth}}",
        r"\toprule",
        r"Context & RASD generation (spec\_steps=4) & "
        r"Target-only generation (spec\_steps=0) \\",
        r"\midrule",
    ]
    txt_lines: list[str] = ["Phase D F5 — qualitative comparison\n",
                            "=" * 78, ""]
    for ctx_key in ["ctx4k", "ctx8k", "ctx128k", "ctx256k", "ctx512k", "ctx1M"]:
        if ctx_key not in pairs:
            continue
        pair = pairs[ctx_key]
        rasd_text = pair.get("RASD",   Path("/dev/null")).read_text(errors="replace") if pair.get("RASD") else "(missing)"
        tgt_text  = pair.get("TARGET", Path("/dev/null")).read_text(errors="replace") if pair.get("TARGET") else "(missing)"
        rasd_ex = _excerpt(rasd_text, args.excerpt_chars)
        tgt_ex  = _excerpt(tgt_text,  args.excerpt_chars)

        # LaTeX-escape ampersands and underscores in the excerpts (simple)
        def _esc(s: str) -> str:
            return (s.replace("\\", r"\textbackslash{}")
                     .replace("&",  r"\&")
                     .replace("_",  r"\_")
                     .replace("%",  r"\%")
                     .replace("#",  r"\#")
                     .replace("$",  r"\$")
                     .replace("{",  r"\{")
                     .replace("}",  r"\}")
                     .replace("~",  r"\textasciitilde{}")
                     .replace("^",  r"\textasciicircum{}"))

        tex_lines.append(
            f"{CTX_LABEL[ctx_key]} & {_esc(rasd_ex)} & {_esc(tgt_ex)} \\\\"
        )

        txt_lines += [
            f"\n--- {CTX_LABEL[ctx_key]} ---",
            f"\n  RASD   : {rasd_ex}",
            f"\n  TARGET : {tgt_ex}",
        ]

    tex_lines += [r"\bottomrule", r"\end{tabular}"]

    out_tex = REPO_ROOT / args.out_tex
    out_txt = REPO_ROOT / args.out_txt
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_lines) + "\n")
    out_txt.write_text("\n".join(txt_lines) + "\n")
    print(f"Wrote {out_tex}")
    print(f"Wrote {out_txt}")
    print()
    print(f"Pairs found:")
    for k, v in sorted(pairs.items()):
        present = [kind for kind in ("RASD", "TARGET") if kind in v]
        print(f"  {k:<10}  {'+'.join(present)}")


if __name__ == "__main__":
    raise SystemExit(main())
