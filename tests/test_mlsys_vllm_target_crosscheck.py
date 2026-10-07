"""The vLLM cross-check is a THROUGHPUT reference; token agreement is descriptive.

RASD runs FP4 target weights and an NF4 KV cache -- that is the configuration the
memory budget and the published speedups were measured under. vLLM 0.6.3 has no
NF4 KV path, so on this model pair it runs bf16 weights and a bf16 cache. Two
engines whose weights and cache are quantized differently will disagree on a
greedy continuation at some position; reading that as "implementation
validation" measures precision and labels it correctness.

So the verdict literal is `NOT_COMPARABLE_PRECISION`, the divergence is reported
with its position, the agreement length up to it and both arms' gaps, and the
stage is decided by whether any row is `unit_matched` for throughput. These tests
drive the production row builder and the production comparison.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mlsys_vllm_baseline.py"

TARGET_REV = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
BOS = 128000
PROMPT = [9906, 1917, 527, 264, 13, 42, 7]
ENGINE_IDS = [BOS] + PROMPT
RASD_IDS = list(range(5000, 5024))        # 24 "emitted" target-only ids


def _load_module():
    spec = importlib.util.spec_from_file_location("_vllm_crosscheck", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_sidecars(tmp: Path) -> Path:
    d = tmp / "tokens"
    d.mkdir(exist_ok=True)
    import hashlib
    common = {
        "doc_id": "pg19_train_0", "context_length": 131072,
        "prompt_tokens": len(PROMPT), "prompt_token_ids": PROMPT,
        "prompt_sha256": hashlib.sha256(
            ",".join(map(str, PROMPT)).encode()).hexdigest(),
        "engine_input_ids": ENGINE_IDS,
    }
    (d / "spec.json").write_text(json.dumps({
        **common, "run_id": "rasd_spec_0", "spec_steps": 4,
        "generated_token_ids": RASD_IDS, "token_gaps": [3.0] * len(RASD_IDS)}))
    (d / "target.json").write_text(json.dumps({
        **common, "run_id": "rasd_target_0", "spec_steps": 0,
        "generated_token_ids": RASD_IDS, "token_gaps": [3.0] * len(RASD_IDS),
        "throughput_tps": 41.5, "weight_precision": "nf4_4bit",
        "kv_dtype": "nf4"}))
    return d


def _args():
    return types.SimpleNamespace(max_new_tokens=len(RASD_IDS),
                                 tensor_parallel_size=8,
                                 matched_max_new_tokens=len(RASD_IDS),
                                 target_revision=TARGET_REV,
                                 draft_revision="")


def _rows(tmp: Path, emitted_ids, *, unit_matched=True):
    """Rows from the production builder, with a fake worker result."""
    mod = _load_module()
    tokens = _write_sidecars(tmp)
    cells = mod.load_rasd_cells(tokens)
    result = {
        "status": "ok", "prompt_tokens": len(PROMPT),
        "prompt_sha256": mod._ids_sha(ENGINE_IDS),
        "prompt_ids_used_sha256": mod._ids_sha(ENGINE_IDS),
        "prompt_ids_verified": "yes",
        "vllm_version": mod.VLLM_PIN if unit_matched else "0.6.2",
        "quantization": "bfloat16", "eos_policy": mod.EOS_POLICY,
        "target_revision": TARGET_REV, "draft_revision": "",
        "output_tokens": len(emitted_ids),
        "end_to_end_wall_s": 1.0, "throughput_tps_end_to_end": 30.0,
        "weight_precision": "bf16", "kv_dtype": "bf16",
        "output_token_ids": list(emitted_ids),
        "output_gaps": [4.0] * len(emitted_ids),
    }
    from_rev = {"meta-llama/Llama-3.1-8B": TARGET_REV}
    return mod, [mod.build_row(cells[0], "meta-llama/Llama-3.1-8B", 131072,
                               ENGINE_IDS, result, attempt_idx=1,
                               config_name="1-default-tp8", log_path="x.log",
                               rope=None, args=_args(), max_model_len=131072,
                               target_revision=TARGET_REV,
                               draft_revision="")]


def _compare(tmp: Path, rows) -> tuple[int, list[dict]]:
    mod = _load_module()
    tokens = tmp / "tokens"
    out = tmp / "cross.csv"
    rc = mod.compare_targets(rows, tokens, out,
                             vllm_sidecars_dir=mod.write_vllm_token_sidecars(
                                 rows, out))
    with out.open() as fh:
        return rc, list(csv.DictReader(fh))


def test_a_token_divergence_is_reported_and_never_fails(tmp_path):
    diverge = list(RASD_IDS)
    diverge[7] = 99999
    mod, rows = _rows(tmp_path, diverge)

    rc, table = _compare(tmp_path, rows)

    assert rc == 0, ("two engines with different weight and KV precision "
                     "diverging on a token is expected; it cannot fail a stage")
    assert table[0]["verdict"] == "NOT_COMPARABLE_PRECISION"
    assert int(table[0]["first_divergence_position"]) == 7
    assert int(table[0]["agreement_prefix"]) == 7
    assert int(table[0]["spec_ids_at_divergence"]) == 99999
    assert int(table[0]["rasd_ids_at_divergence"]) == RASD_IDS[7]
    assert table[0]["gap_at_divergence_spec"] not in ("", None)


def test_full_agreement_is_reported_as_the_compared_prefix(tmp_path):
    mod, rows = _rows(tmp_path, RASD_IDS)

    rc, table = _compare(tmp_path, rows)

    assert rc == 0
    assert table[0]["verdict"] == "NOT_COMPARABLE_PRECISION", (
        "agreement under different precision is still not a correctness claim")
    assert table[0]["first_divergence_position"] == ""
    assert int(table[0]["agreement_prefix"]) == len(RASD_IDS)


def test_both_engines_precision_is_recorded(tmp_path):
    mod, rows = _rows(tmp_path, RASD_IDS)

    _rc, table = _compare(tmp_path, rows)

    assert table[0]["vllm_weight_precision"] == "bf16"
    assert table[0]["vllm_kv_dtype"] == "bf16"
    assert table[0]["rasd_weight_precision"] == "nf4_4bit"
    assert table[0]["rasd_kv_dtype"] == "nf4", (
        "the table must show that the two engines did not run the same cache")


def test_a_stage_with_no_unit_matched_row_fails(tmp_path):
    mod, rows = _rows(tmp_path, RASD_IDS, unit_matched=False)

    rc, table = _compare(tmp_path, rows)

    assert rc != 0, ("the stage's only claim is the throughput reference, so a "
                     "stage whose rows are not comparable has no claim at all")
    assert table[0]["verdict"] == "NOT_UNIT_MATCHED"
    assert table[0]["unit_matched"] == "no"


def test_a_missing_pair_is_reported_and_does_not_fail_the_reference(tmp_path):
    """The descriptive comparison can be unavailable without costing the claim.

    The stage's claim is a THROUGHPUT reference, which needs the document, the
    matched `max_new_tokens` and the unit match -- not the RASD token sidecar.
    So a cell with no target-only partner is reported as NO_PAIR, its tokens are
    not compared, and the throughput reference still stands. (This is the
    opposite of the losslessness stages, where a missing pair is a hard failure:
    there, the pairing is the claim.)
    """
    mod, rows = _rows(tmp_path, RASD_IDS)
    # Drop the target-only sidecar, keeping only the speculative one.
    (tmp_path / "tokens" / "target.json").unlink()

    rc, table = _compare(tmp_path, rows)

    assert table[0]["verdict"] == "NO_PAIR", table[0]
    assert table[0]["unit_matched"] == "yes"
    assert rc == 0, "the throughput reference does not depend on the token pair"
    assert table[0]["agreement_prefix"] in ("", None), (
        "a cell with no partner must not report an agreement length")


def test_the_verdict_never_uses_the_losslessness_words(tmp_path):
    """`LOSSLESS`/`MISMATCH` would assert that the comparison is meaningful."""
    diverge = list(RASD_IDS)
    diverge[3] = 4242
    mod, rows = _rows(tmp_path, diverge)
    _rc, table = _compare(tmp_path, rows)

    v = table[0]["verdict"]
    assert "LOSSLESS" not in v and "MISMATCH" not in v and "TIE" not in v, v
    assert "precision" in (table[0]["detail"] + v).lower() or v == \
        "NOT_COMPARABLE_PRECISION"
