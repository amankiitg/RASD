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
            "n_acc", "draft_tokens", "accepted",
        }

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
        lines = RASD_INF_SRC.splitlines()
        append_idx = next(
            (i for i, ln in enumerate(lines)
             if "per_token_trace.append(" in ln),
            None,
        )
        assert append_idx is not None, (
            "C13 regression: per_token_trace.append( … ) not present — "
            "trace not being recorded inside the verify loop"
        )
        # Walk backwards up to 8 lines for the cfg.log_per_token guard
        found = False
        for i in range(append_idx - 1, max(0, append_idx - 8), -1):
            if re.match(r"\s*if cfg\.log_per_token:", lines[i]):
                found = True
                break
        assert found, (
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
