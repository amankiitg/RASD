"""GPU-free stand-in for `scripts/mlsys_coherence_gate.py`.

The gate's DECISION layer is what the rehearsal needs to exercise, and that layer
is pure: which reference a candidate is judged against (`reference_context`, i.e.
an extension is judged against the declared native baseline, a control against
its own context), whether a declared baseline exists at all, the pass rule in
`verdict()`, and the CSV schema. None of that needs weights.

So this module replaces the one GPU-dependent function -- `run_candidate`, which
loads a model and measures -- and then calls the REAL `main()`. Everything
downstream of the measurement is the production code path: the same fields, the
same baseline selection, the same tolerance, the same file.

THE SYNTHETIC MEASUREMENT RULE
------------------------------
The stub has to decide, per candidate, whether the simulated target is coherent.
It derives that from the configuration alone -- it never reads the file's
`expect:` field, which the rehearsal then checks the real decision layer against:

  * `rope_type: none`                        -> the model's own rope: coherent
  * a declared anchor >= the context length  -> the historical mis-anchoring
    (max_position_embeddings = context), which is the ARM4-f2 / Llama-2 bug
  * yarn with NO declared anchor             -> the unanchored rebase: the same
    bug reached by omission
  * anything else                            -> coherent, with a small penalty
    proportional to the factor

Those are stand-ins for measurement, not measurements. They are deliberately
coarse; the rehearsal asserts that the real gate code turns these measured rows
into the verdicts each candidate file declares.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
# scripts/_real_coherence_gate.py in the sandbox.
REAL = HERE.parent / "_real_coherence_gate.py"


def _load_real():
    spec = importlib.util.spec_from_file_location("_real_coherence_gate", REAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_real()

# The rope each target ships with. Llama-3.1's block IS a llama3-type scaling
# with factor 8 anchored on its 8192 training window; Llama-2's is plain rope on
# 4096.
SHIPPED = {
    "meta-llama/Llama-3.1-8B": {"rope_type": "llama3", "factor": 8.0,
                                "anchor": 8192},
    "meta-llama/Llama-2-7b-hf": {"rope_type": "none", "factor": None,
                                 "anchor": 4096},
}
BASE_PPL = 14.0
DEGENERATE_MULTIPLIER = 8.0


class _FakeTokenizer:
    """Only `pad_token` / `eos_token` are read on this path."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *_a, **_kw):
        return cls()


def _misanchored(cand: dict) -> bool:
    ctx = int(cand.get("context_length", 0))
    rtype = str(cand.get("rope_type") or "none").lower()
    anchor = cand.get("rope_anchor_base")
    if rtype in ("none", ""):
        return False
    if anchor not in (None, "", "None"):
        return int(anchor) >= ctx
    return rtype == "yarn"


def _sample_hash(cand: dict, kind: str) -> str:
    """The hash of the window this candidate scored.

    In the real gate the window is drawn from the metadata by (context, seed)
    alone -- the model does not enter the draw -- so two rows with the same
    context and seed score the SAME document, offset, prompt and continuation,
    and a row with a different seed scores a different one. Mirroring that is
    what lets the rehearsal exercise the pairing rule: a reference the config
    failed to align shows up as an 'unpaired' row rather than passing silently.
    """
    key = f"{kind}|{cand.get('context_length')}|{cand.get('seed', 42)}"
    return hashlib.sha256(key.encode()).hexdigest()


def _ppl(cand: dict) -> float:
    rtype = str(cand.get("rope_type") or "none").lower()
    factor = float(cand.get("rope_factor") or 1.0)
    if rtype in ("none", ""):
        return BASE_PPL
    if _misanchored(cand):
        return BASE_PPL * DEGENERATE_MULTIPLIER
    # A small, monotone penalty so a bigger factor is not silently free.
    return BASE_PPL * (1.0 + 0.05 * factor / 32.0)


def run_candidate(cand: dict, tok, meta_path: str, out_dir: pathlib.Path):
    name = cand["name"]
    ctx = int(cand["context_length"])
    rtype = str(cand.get("rope_type") or "none").lower()
    bad = _misanchored(cand)
    row = {
        "candidate": name,
        "target_model_name": cand["target_model_name"],
        "context_length": ctx,
        "seed": cand.get("seed", 42),
        "rope_type": cand.get("rope_type"),
        "rope_factor": cand.get("rope_factor"),
        "rope_anchor_base": cand.get("rope_anchor_base"),
        "reference_context": cand.get("reference_context"),
        "target_revision": cand.get("target_revision"),
        "native_baseline": bool(cand.get("native_baseline", False)),
        "role": cand.get("role", ""),
        "expect": cand.get("expect", ""),
        # Carried through so the CSV says WHICH KIND of reference a row is: the
        # unscaled 32k/128k rows are OOD references, not in-distribution ones.
        "reference_role": cand.get("reference_role", ""),
        # The builder's audit trail, as run_candidate records it.
        "config_max_position_embeddings": cand.get("rope_anchor_base") or ctx,
        "config_rope_scaling": json.dumps(
            {"rope_type": rtype} if rtype != "none" else None),
        # The rope assertion, with the label vocabulary assert_effective_rope
        # produces: "declared" when the build matched the declaration, and
        # "anchor_on_context" for the mis-anchored build.
        "effective_rope_match": "anchor_on_context" if bad else "declared",
        "effective_rope_maxerr": 1.0 if bad else 0.0,
        "effective_rope_matches_intent": not bad,
        "native_window": SHIPPED[cand["target_model_name"]]["anchor"],
        "inv_freq_first": "1.0",
        "inv_freq_last": "1e-05",
        "slowest_channel_stretch": 1.0 if not bad else float(ctx),
        "prompt_tokens": ctx,
        # The sample identity: the same (context, seed) scores the same window,
        # so the gate can tell a paired contrast from two different samples.
        "prompt_sha256": _sample_hash(cand, "prompt"),
        "continuation_sha256": _sample_hash(cand, "continuation"),
        "seed": cand.get("seed", 42),
        "ppl_continuation": round(_ppl(cand), 4),
    }
    if bad:
        # The ARM4 f2 shape: the target emits EOS almost immediately and the
        # generation is blank/repetitive.
        row.update({"early_eos": True, "eos_at": 0, "gen_chars": 0,
                    "gen_blank_share": 0.35, "gen_alpha_share": 0.05,
                    "gen_repeat_share": 0.5})
    else:
        row.update({"early_eos": False, "eos_at": "", "gen_chars": 2400,
                    "gen_blank_share": 0.0, "gen_alpha_share": 0.82,
                    "gen_repeat_share": 0.05})
    row["status"] = "ok"
    row["error"] = ""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gen_{name}.txt").write_text(
        "" if bad else "rehearsal stub continuation text\n" * 40)
    return row


def main() -> int:
    gate.run_candidate = run_candidate
    gate.AutoTokenizer = _FakeTokenizer
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
