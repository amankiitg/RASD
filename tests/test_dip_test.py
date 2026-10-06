"""Tests for src/analysis/dip_test.py — Hartigan dip test wrapper.

These establish that the wiring around `diptest` behaves correctly:
unimodal samples are not rejected, clearly bimodal samples are, the
statistic respects its 1/4 bound, and degenerate inputs (too few rounds,
all-identical values) are flagged rather than crashing a whole analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.dip_test import (
    aggregate_by_context,
    dip_over_trace_dir,
    dip_test,
)

REPO = Path(__file__).resolve().parent.parent


class TestDipTest:
    def test_unimodal_normal_is_not_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.5, 0.1, size=300)
        out = dip_test(x)
        assert out["dip"] < 0.06
        assert out["p_value"] > 0.05
        assert out["reject_unimodal"] is False

    def test_separated_bimodal_is_rejected(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(0.1, 0.02, 150),
                            rng.normal(0.9, 0.02, 150)])
        out = dip_test(x)
        assert out["dip"] > 0.2
        assert out["p_value"] < 0.01
        assert out["reject_unimodal"] is True

    def test_zero_plus_nonzero_spike_is_rejected(self):
        """The exact pattern the reviewer doubted: a spike at 0 next to a
        nonzero mode. The dip test should confirm bimodality rather than
        the claim resting on the zero/nonzero split."""
        rng = np.random.default_rng(7)
        x = np.concatenate([np.zeros(60), rng.normal(0.85, 0.05, 60)])
        out = dip_test(x)
        assert out["reject_unimodal"] is True
        assert out["p_value"] < 0.01

    @pytest.mark.parametrize("n", [10, 50, 500])
    def test_dip_respects_quarter_bound(self, n):
        rng = np.random.default_rng(n)
        for _ in range(5):
            x = rng.random(n)
            assert 0.0 <= dip_test(x)["dip"] <= 0.25 + 1e-9

    def test_too_few_points_is_degenerate_not_an_error(self):
        out = dip_test([0.0, 0.5, 1.0])
        assert out["degenerate"] is True
        assert out["dip"] == 0.0
        assert out["p_value"] == 1.0

    def test_all_identical_is_degenerate(self):
        out = dip_test([0.5] * 40)
        assert out["degenerate"] is True
        assert out["reject_unimodal"] is False

    def test_nan_values_are_dropped(self):
        out = dip_test([0.0, np.nan, 1.0, 0.5, 0.25, np.nan])
        assert out["n"] == 4
        assert not out["degenerate"]

    def test_bootstrap_pvalue_is_seed_reproducible(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0.1, 0.02, 60),
                            rng.normal(0.9, 0.02, 60)])
        a = dip_test(x, boot_pval=True, n_boot=500, seed=3)
        b = dip_test(x, boot_pval=True, n_boot=500, seed=3)
        assert a["p_value"] == b["p_value"]
        assert a["p_method"] == "uniform_bootstrap"
        assert a["dip"] == pytest.approx(b["dip"])

    def test_alpha_threshold_controls_the_flag(self):
        rng = np.random.default_rng(5)
        x = np.concatenate([np.zeros(40), rng.normal(0.9, 0.03, 40)])
        # a p-value can never be < 0, so a 0-level test never rejects;
        # a lenient level does. (This sample's p-value is far below 1e-3,
        # so 0.001 was not a usable "lenient" threshold.)
        assert dip_test(x, alpha=0.0)["reject_unimodal"] is False
        assert dip_test(x, alpha=0.5)["reject_unimodal"] is True
        # and the flag is exactly p_value < alpha
        out = dip_test(x, alpha=0.5)
        assert out["reject_unimodal"] == (out["p_value"] < 0.5)


def _write_trace(path: Path, run_id: str, n_acc, gamma=4):
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{run_id}.jsonl").open("w") as f:
        for i, n in enumerate(n_acc):
            f.write(json.dumps({
                "round_idx": i, "global_pos_start": i, "spec_steps": gamma,
                "n_acc": int(n), "draft_tokens": [0] * gamma,
                "accepted": [j < n for j in range(gamma)],
            }) + "\n")


class TestDipOverTraceDir:
    def test_one_row_per_trace(self, tmp_path):
        d = tmp_path / "per_token"
        _write_trace(d, "M4_ctx128k_s42", [0, 4, 0, 4, 0, 4, 1, 2])
        _write_trace(d, "M4_ctx128k_s123", [0, 4, 0, 4, 0, 4, 1, 2])
        out = dip_over_trace_dir(d)
        assert len(out) == 2
        assert set(out["run_id"]) == {"M4_ctx128k_s42", "M4_ctx128k_s123"}
        assert (out["context_length"] == 131072).all()
        assert out["n_rounds"].tolist() == [8, 8]

    def test_empty_dir_returns_empty_frame(self, tmp_path):
        out = dip_over_trace_dir(tmp_path)
        assert out.empty
        assert "dip" in out.columns


class TestAggregateByContext:
    def test_seed_count_is_reported_honestly(self, tmp_path):
        """A 1-seed context must not aggregate as though it were 3-seed."""
        d = tmp_path / "per_token"
        _write_trace(d, "M4_ctx128k_s42", [0, 4, 0, 4, 2, 1, 0, 3])
        one_seed = aggregate_by_context(dip_over_trace_dir(d))
        row = one_seed[one_seed["context_length"] == 131072].iloc[0]
        assert row["n_seeds"] == 1

        _write_trace(d, "M4_ctx128k_s123", [0, 4, 0, 4, 2, 1, 0, 3])
        _write_trace(d, "M4_ctx128k_s456", [0, 4, 0, 4, 2, 1, 0, 3])
        three = aggregate_by_context(dip_over_trace_dir(d))
        row3 = three[three["context_length"] == 131072].iloc[0]
        assert row3["n_seeds"] == 3
        # Rejecting SEEDS can never exceed the number of seeds. The old
        # single `n_reject` column summed over runs and could report
        # impossible values like "5/3 seeds" when a context had several
        # variants per seed.
        assert row3["n_seeds_reject"] <= row3["n_seeds"]
        assert row3["n_runs_reject"] <= row3["n_runs"]
        assert row3["n_runs"] == 3 and row3["n_seeds"] == 3
        # three identical seeds -> zero-width CI
        assert row3["dip_ci_lo"] == pytest.approx(row3["dip_ci_hi"])

    def test_runs_and_seeds_are_counted_separately(self, tmp_path):
        """Two variants of one seed must not inflate the seed count."""
        d = tmp_path / "per_token"
        # Same seed, two different variants at the same context (as the
        # 64k NF4-vs-bf16 isolation produces).
        _write_trace(d, "BF16_ctx64k_nf4_s42", [0, 4, 0, 4, 2, 1, 0, 3])
        _write_trace(d, "BF16_ctx64k_bf16_s42", [0, 4, 0, 4, 2, 1, 0, 3])
        agg = aggregate_by_context(dip_over_trace_dir(d))
        row = agg[agg["context_length"] == 65536].iloc[0]
        assert row["n_seeds"] == 1        # ONE seed
        assert row["n_runs"] == 2         # TWO runs
        assert row["n_seeds_reject"] <= 1

    def test_missing_context_is_not_a_context(self, tmp_path):
        """Rows without a context must not be bucketed into a bogus row."""
        d = tmp_path / "per_token"
        _write_trace(d, "M4_ctx128k_s42", [0, 4, 0, 4, 2, 1, 0, 3])
        _write_trace(d, "canary_s42", [0, 4, 0, 4, 2, 1, 0, 3])
        agg = aggregate_by_context(dip_over_trace_dir(d))
        assert agg["context_length"].notna().all()
        assert set(agg["context_length"]) == {131072.0}

    def test_empty_input(self):
        assert aggregate_by_context(pd.DataFrame()).empty
