"""Tests for the M4 mentor-required sidecar instrumentation.

Covers C12 (TTFT) and C13 (per-position acceptance trace) — both default
off so M3 replay stays byte-identical when flags are unchanged.

The runtime paths require multi-GPU NCCL; we test the pure helper
function functionally, the schema, and source-inspect the integration
points in `generate()` to lock in the wiring.
"""

import re
from pathlib import Path
import torch

from src.models.rasd_inference import RASDConfig, _build_per_token_record

REPO_ROOT = Path(__file__).resolve().parent.parent
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()


# ---------------------------------------------------------------------------
# C13 — per-position record helper (pure function)
# ---------------------------------------------------------------------------

class TestPerTokenRecord:
    def test_schema_keys(self):
        """The .jsonl schema is the contract Figure 4 reads from. Lock it."""
        draft_seq = torch.tensor([[10, 20, 30, 40]])
        accepted  = torch.tensor([[True, True, False, False]])
        rec = _build_per_token_record(
            round_idx=3, global_pos_start=128, spec_steps=4,
            n_acc=2, draft_seq=draft_seq, accepted=accepted,
        )
        assert set(rec.keys()) == {
            "round_idx", "global_pos_start", "spec_steps",
            "n_acc", "draft_tokens", "accepted", "ended_on_eos",
        }

    def test_ended_on_eos_present_and_false_by_default(self):
        """The EOS flag must exist on EVERY record, not just the last one, so
        analysis never conflates "key absent" with "round did not end on EOS".
        The verify loop flips it in place on the terminating round."""
        rec = _build_per_token_record(
            round_idx=0, global_pos_start=0, spec_steps=4, n_acc=4,
            draft_seq=torch.tensor([[1, 2, 3, 4]]),
            accepted=torch.tensor([[True, True, True, True]]),
        )
        assert "ended_on_eos" in rec
        assert rec["ended_on_eos"] is False

    def test_verify_loop_sets_ended_on_eos_before_break(self):
        """Source-inspect the integration: the flag must be set on the trace
        record inside the same branch that breaks the verify loop, and the
        break must stay gated on `not cfg.ignore_eos` (B3)."""
        from tests.source_guard_utils import (
            assignment_precedes_break, call_gated_by,
        )
        # Ordering is the property: the flag is flipped on the terminating round
        # and THEN the loop breaks. Asserting that by adjacency to a regex-captured
        # `break` broke as soon as an unrelated statement was inserted between
        # them, which reported a regression that had not happened.
        assert assignment_precedes_break(
            RASD_INF_SRC, 'per_token_trace[-1]["ended_on_eos"] = True',
            guard_fragment="cfg.log_per_token",
        ), (
            "ended_on_eos must be flipped on the terminating round before break, "
            "and the flip must be gated on cfg.log_per_token so non-traced runs "
            "stay byte-identical"
        )

    def test_ignore_eos_suppresses_the_flag(self):
        """With ignore_eos set the break never fires, so generation runs to
        max_new_tokens and every record must keep ended_on_eos=False."""
        assert re.search(
            r"if \(not cfg\.ignore_eos\) and \(cur_token == self\.tokenizer\.eos_token_id\)\.all\(\):",
            RASD_INF_SRC,
        ), "B3 regression: the EOS break is no longer gated on ignore_eos"

    def test_values_round_trip(self):
        draft_seq = torch.tensor([[7, 8, 9, 10]])
        accepted  = torch.tensor([[True, False, False, False]])
        rec = _build_per_token_record(
            round_idx=0, global_pos_start=64, spec_steps=4,
            n_acc=1, draft_seq=draft_seq, accepted=accepted,
        )
        assert rec["round_idx"] == 0
        assert rec["global_pos_start"] == 64
        assert rec["spec_steps"] == 4
        assert rec["n_acc"] == 1
        assert rec["draft_tokens"] == [7, 8, 9, 10]
        assert rec["accepted"] == [True, False, False, False]

    def test_jsonl_serializable(self):
        """Each record must JSON-encode without TypeError so it can land
        in a .jsonl sidecar without a custom encoder."""
        import json
        rec = _build_per_token_record(
            round_idx=0, global_pos_start=10, spec_steps=2, n_acc=2,
            draft_seq=torch.tensor([[1, 2]]),
            accepted=torch.tensor([[True, True]]),
        )
        line = json.dumps(rec)
        assert json.loads(line) == rec

    def test_full_acceptance(self):
        """n_acc == k means every draft token was accepted."""
        rec = _build_per_token_record(
            round_idx=5, global_pos_start=200, spec_steps=4, n_acc=4,
            draft_seq=torch.tensor([[1, 2, 3, 4]]),
            accepted=torch.tensor([[True, True, True, True]]),
        )
        assert rec["n_acc"] == 4
        assert all(rec["accepted"])

    def test_zero_acceptance(self):
        """n_acc == 0 means the very first draft token was rejected."""
        rec = _build_per_token_record(
            round_idx=2, global_pos_start=20, spec_steps=4, n_acc=0,
            draft_seq=torch.tensor([[1, 2, 3, 4]]),
            accepted=torch.tensor([[False, False, False, False]]),
        )
        assert rec["n_acc"] == 0
        assert not any(rec["accepted"])

    def test_native_python_types(self):
        """Values must be plain ints/bools/lists, not torch scalars (which
        json.dumps refuses without a custom encoder)."""
        rec = _build_per_token_record(
            round_idx=0, global_pos_start=0, spec_steps=2, n_acc=1,
            draft_seq=torch.tensor([[100, 200]]),
            accepted=torch.tensor([[True, False]]),
        )
        assert isinstance(rec["round_idx"], int)
        assert isinstance(rec["n_acc"], int)
        assert isinstance(rec["accepted"][0], bool)
        assert isinstance(rec["draft_tokens"][0], int)


# ---------------------------------------------------------------------------
# C12 — TTFT integration
# ---------------------------------------------------------------------------

class TestTTFT:
    def test_metrics_dict_includes_ttft(self):
        assert re.search(
            r'"ttft_ms"\s*:\s*\(t_first_token\s*-\s*t_start\)\s*\*\s*1000',
            RASD_INF_SRC,
        ), "C12 regression: ttft_ms missing from metrics dict in generate()"

    def test_first_token_timestamp_captured_after_initial_sample(self):
        """t_first_token must be captured after the seed sample, not before
        — otherwise we'd be measuring entry-time, not real prefill+sample."""
        lines = RASD_INF_SRC.splitlines()
        # Find the seed-sample line (the one assigning generated = [cur_token])
        seed_idx = next(
            i for i, ln in enumerate(lines)
            if "generated  = [cur_token]" in ln
        )
        # Find t_first_token assignment
        ttft_idx = next(
            i for i, ln in enumerate(lines)
            if "t_first_token = time.perf_counter()" in ln
        )
        assert ttft_idx > seed_idx, (
            "C12 regression: t_first_token captured BEFORE the first token "
            "was sampled — TTFT metric would be wrong"
        )

    def test_ttft_uses_perf_counter(self):
        """Use the same clock as t_start (time.perf_counter), not time.time()."""
        assert "t_first_token = time.perf_counter()" in RASD_INF_SRC, (
            "C12 regression: TTFT must use time.perf_counter() to match t_start"
        )


# ---------------------------------------------------------------------------
# C13 — per-position trace integration
# ---------------------------------------------------------------------------

class TestPerTokenTraceIntegration:
    def test_log_per_token_config_default_off(self):
        """Default must be False so M3 replay stays byte-identical when
        the flag is not set."""
        cfg = RASDConfig()
        assert cfg.log_per_token is False, (
            "C13 regression: log_per_token default flipped to True — "
            "would change M3 replay output"
        )

    def test_trace_append_inside_log_guard(self):
        """The append into per_token_trace must be inside `if cfg.log_per_token`
        otherwise a small allocation runs every round at no benefit."""
        from tests.source_guard_utils import call_gated_by
        assert "per_token_trace.append(" in RASD_INF_SRC, (
            "C13 regression: per_token_trace.append( … ) not present — "
            "trace not being recorded inside the verify loop"
        )
        # Structural, not a fixed backwards window: an insertion between the
        # guard and the call is not a regression, an unguarded call is.
        assert call_gated_by(
            RASD_INF_SRC, "per_token_trace.append(", "cfg.log_per_token"
        ), (
            "C13 regression: per_token_trace.append is not gated by "
            "`if cfg.log_per_token:` — adds allocation cost when disabled"
        )

    def test_metrics_contains_trace_when_enabled(self):
        """When cfg.log_per_token is True, metrics dict gets per_token_trace."""
        assert re.search(
            r'metrics\["per_token_trace"\]', RASD_INF_SRC,
        ), "C13 regression: metrics['per_token_trace'] not populated"

    def test_only_rank_zero_returns_trace(self):
        """Avoid duplicate sidecars: ranks 1..N-1 return None for the trace."""
        # All ranks lockstep so the trace is identical; non-zero ranks
        # returning None prevents downstream callers from accidentally
        # writing the same trace 8 times.
        assert re.search(
            r"per_token_trace if self\._rank == 0 else None",
            RASD_INF_SRC,
        ), "C13 regression: non-zero ranks not nulling per_token_trace"
