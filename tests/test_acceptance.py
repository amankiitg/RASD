"""Tests for src/analysis/acceptance.py — per-round acceptance accounting.

These pin the distinction the MLSys reviewer raised (AbH52 #1): the
reported `acceptance_rate` is the per-round accepted-prefix fraction
SUM(n_acc)/(R*gamma), and the i.i.d. per-token parameter is a DERIVED
quantity that only coincides with it under memorylessness.

The last test runs the accounting against the real seed-42 traces
committed in results/final/per_token/ when they are present, so the
schemas cannot silently drift apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.acceptance import (
    _iid_pmf,
    iid_alpha_for_mean,
    iid_ks_distance,
    load_trace,
    parse_run_id,
    per_round_alpha,
    summarize_trace,
    verify_csv_acceptance,
)

REPO = Path(__file__).resolve().parent.parent
REAL_TRACES = REPO / "results" / "final" / "per_token"


def _trace(n_acc_list, gamma=4):
    return [
        {"round_idx": i, "global_pos_start": i, "spec_steps": gamma,
         "n_acc": n, "draft_tokens": [0] * gamma,
         "accepted": [j < n for j in range(gamma)]}
        for i, n in enumerate(n_acc_list)
    ]


class TestParseRunId:
    @pytest.mark.parametrize("run_id,ctx,seed", [
        ("M4_ctx128k_s42", 131072, 42),
        ("M4_ctx1M_s456", 1048576, 456),
        ("RASD_ctx4k_pg19_phaseD_s42", 4096, 42),
        ("P35D_ctx1M_pg19_s42", 1048576, 42),
        ("mlsys_arm2_canary_s42", None, 42),
    ])
    def test_parses_both_id_dialects(self, run_id, ctx, seed):
        assert parse_run_id(run_id) == (ctx, seed)


class TestPerRoundAlpha:
    def test_alpha_is_n_acc_over_gamma(self):
        got = per_round_alpha(_trace([0, 2, 4], gamma=4))
        np.testing.assert_allclose(got, [0.0, 0.5, 1.0])

    def test_skips_zero_gamma_rounds(self):
        trace = _trace([2], gamma=4)
        trace.append({"round_idx": 9, "spec_steps": 0, "n_acc": 0})
        assert per_round_alpha(trace).size == 1


class TestSummarize:
    def test_alpha_round_equals_total_accepted_over_total_proposed(self):
        trace = _trace([0, 1, 4, 3], gamma=4)
        s = summarize_trace(trace)
        assert s["alpha_round"] == pytest.approx(8 / 16)
        # the identity the module docstring claims
        assert s["alpha_round"] == pytest.approx(
            s["total_accepted"] / s["total_proposed"])

    def test_zero_and_full_accept_fractions(self):
        s = summarize_trace(_trace([0, 0, 4, 2], gamma=4))
        assert s["p_zero"] == pytest.approx(0.5)
        assert s["frac_full_accept"] == pytest.approx(0.25)

    def test_target_only_trace_is_nan_not_error(self):
        s = summarize_trace([])
        assert s["n_rounds"] == 0
        assert np.isnan(s["alpha_round"])
        assert np.isnan(s["alpha_iid"])


class TestIidModel:
    """E[N] must equal sum_{i=1..gamma} a^i and invert exactly."""

    @pytest.mark.parametrize("gamma", [1, 2, 4, 8])
    @pytest.mark.parametrize("a", [0.01, 0.1, 0.25, 0.5, 0.9])
    def test_mean_matches_closed_form(self, gamma, a):
        mean = float((_iid_pmf(a, gamma) * np.arange(gamma + 1)).sum())
        assert mean == pytest.approx(sum(a ** i for i in range(1, gamma + 1)))

    @pytest.mark.parametrize("gamma", [2, 4, 8])
    @pytest.mark.parametrize("a", [0.05, 0.2, 0.5, 0.8])
    def test_inversion_is_exact(self, gamma, a):
        mean = float((_iid_pmf(a, gamma) * np.arange(gamma + 1)).sum())
        assert iid_alpha_for_mean(mean, gamma) == pytest.approx(a, abs=1e-9)

    def test_pmf_sums_to_one(self):
        assert _iid_pmf(0.3, 4).sum() == pytest.approx(1.0)

    def test_iid_alpha_exceeds_alpha_round_when_acceptance_is_bursty(self):
        """Bursty acceptance (spike at 0 plus spike at gamma) needs a much
        higher i.i.d. alpha to reproduce the same mean — the reviewer's point."""
        trace = _trace([4, 4, 0, 0], gamma=4)   # mean n_acc = 2
        s = summarize_trace(trace)
        assert s["alpha_round"] == pytest.approx(0.5)
        assert s["alpha_iid"] > s["alpha_round"]

    def test_ks_small_for_data_from_the_model(self):
        """Sample genuinely from the memoryless model -> small KS distance."""
        rng = np.random.default_rng(0)
        alpha, gamma, n = 0.6, 4, 20000
        u = rng.random(n)
        # simulate N = length of leading run of successes, capped at gamma
        n_acc = np.zeros(n, dtype=int)
        for i in range(n):
            k = 0
            while k < gamma and rng.random() < alpha:
                k += 1
            n_acc[i] = k
        assert iid_ks_distance(n_acc.astype(float), gamma) < 0.03

    def test_ks_large_for_spike_at_zero(self):
        """Half zeros and half full accepts is not memoryless.

        With gamma=4 the fitted alpha is ~0.74 (E[N]=2), so the model puts
        only ~0.26 mass at n_acc=0 against 0.5 observed — a KS distance of
        ~0.24, i.e. an order of magnitude above the ~0.02 seen for data
        actually drawn from the model.
        """
        n_acc = np.array([0.0] * 50 + [4.0] * 50)
        ks = iid_ks_distance(n_acc, 4)
        assert ks > 0.2
        assert ks > 5 * 0.03


class TestVerifyCsvAcceptance:
    def _write_traces(self, tmp_path, run_id, n_acc):
        d = tmp_path / "per_token"
        d.mkdir()
        import json
        with (d / f"{run_id}.jsonl").open("w") as f:
            for rec in _trace(n_acc):
                f.write(json.dumps(rec) + "\n")
        return d

    def test_ok_when_csv_matches_trace(self, tmp_path):
        import pandas as pd
        d = self._write_traces(tmp_path, "M4_ctx128k_s42", [0, 2, 4, 2])
        csv = tmp_path / "r.csv"
        pd.DataFrame([{"run_id": "M4_ctx128k_s42",
                       "acceptance_rate": 0.5}]).to_csv(csv, index=False)
        out = verify_csv_acceptance(csv, d)
        assert bool(out.loc[0, "ok"]) is True
        assert out.loc[0, "abs_diff"] == pytest.approx(0.0)

    def test_not_ok_when_csv_reports_the_iid_parameter(self, tmp_path):
        """If a table accidentally stores alpha_iid, the check must fail."""
        import pandas as pd
        d = self._write_traces(tmp_path, "M4_ctx128k_s42", [4, 4, 0, 0])
        csv = tmp_path / "r.csv"
        pd.DataFrame([{"run_id": "M4_ctx128k_s42",
                       "acceptance_rate": 0.9}]).to_csv(csv, index=False)
        out = verify_csv_acceptance(csv, d)
        assert bool(out.loc[0, "ok"]) is False

    def test_missing_trace_is_unverified_not_silently_ok(self, tmp_path):
        import pandas as pd
        d = tmp_path / "per_token"
        d.mkdir()
        csv = tmp_path / "r.csv"
        pd.DataFrame([{"run_id": "TARGET_x_s42",
                       "acceptance_rate": 0.0}]).to_csv(csv, index=False)
        out = verify_csv_acceptance(csv, d)
        assert out.loc[0, "ok"] is None
        assert "no trace" in out.loc[0, "note"]

    def test_tolerates_4dp_csv_rounding(self, tmp_path):
        """The CSV stores acceptance_rate rounded to 4 dp, so a correctly
        accounted row can differ from the trace by up to 5e-5 and must
        still pass."""
        import pandas as pd
        d = self._write_traces(tmp_path, "M4_ctx128k_s42", [0, 1, 4, 3])
        truth = 8 / 16
        csv = tmp_path / "r.csv"
        pd.DataFrame([{"run_id": "M4_ctx128k_s42",
                       "acceptance_rate": round(truth, 4)}]).to_csv(csv, index=False)
        out = verify_csv_acceptance(csv, d)
        assert bool(out.loc[0, "ok"]) is True
        assert out.loc[0, "abs_diff"] < 1e-4


@pytest.mark.skipif(not REAL_TRACES.is_dir(), reason="no committed traces")
class TestAgainstCommittedTraces:
    def test_real_traces_parse_and_summarize(self):
        files = sorted(REAL_TRACES.glob("*.jsonl"))
        assert files, "expected committed seed-42 traces"
        for p in files:
            trace = load_trace(p)
            assert trace, p
            s = summarize_trace(trace)
            assert 0.0 <= s["alpha_round"] <= 1.0
            assert 0.0 <= s["p_zero"] <= 1.0
            # i.i.d. fit must be >= the per-round value on these bursty
            # traces: the memoryless model needs a higher alpha to match
            # the same mean when acceptance is clumped at 0 and gamma.
            assert s["alpha_iid"] >= s["alpha_round"] - 1e-9
