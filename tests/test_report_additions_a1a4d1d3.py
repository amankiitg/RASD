"""Report-gate additions A1, A4, D1-D3.

These pin the properties that make the new analysis claims defensible: that a
"3-seed" row really has three seeds, that an unverifiable row is not silently
reported as a pass, and that a null difference is never called equivalence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ana():
    return _load("mlsys_analysis", "scripts/mlsys_analysis.py")


@pytest.fixture(scope="module")
def rep():
    return _load("mlsys_report_additions", "scripts/mlsys_report_additions.py")


class TestA1SeedCoverage:
    def test_multiseed_files_are_inputs_not_outputs(self, ana):
        # Treating raw run output as a derived aggregate excluded the Phase 2
        # and Phase 3 cells from the aggregation entirely.
        assert "pg19_multiseed.csv" not in ana.OUTPUT_NAMES
        assert "bf16_draft_isolation.csv" not in ana.OUTPUT_NAMES

    def test_series_maps_historical_labels_onto_multiseed_labels(self, ana):
        # The seed-42 rows carry the ORIGINAL Phase-D labels; joining on
        # run_id alone leaves the seeds in two buckets and coverage still
        # reads n=2. The series key is what actually unites them.
        df = pd.DataFrame([
            # new run: seeds 123/456
            {"group": "PG19_MULTISEED", "level_id": "PG19_ctx4k",
             "run_id": "PG19_ctx4k_s123", "context_length": 4096,
             "spec_steps": 4, "seed": 123},
            # historical seed-42 for the SAME series, different label
            {"group": "M4", "level_id": "RASD_ctx4k_pg19_phaseD",
             "run_id": "RASD_ctx4k_pg19_phaseD_s42", "context_length": 4096,
             "spec_steps": 4, "seed": 42},
        ])
        out = ana.add_series(df)
        assert out["series"].nunique() == 1

    def test_series_keeps_target_apart_from_spec(self, ana):
        # A baseline exists to divide by, not to pool with. The historical
        # TARGET_*_pg19_phaseD row must not be relabelled as a spec cell.
        df = pd.DataFrame([
            {"group": "PG19_MULTISEED", "level_id": "PG19_ctx4k",
             "run_id": "PG19_ctx4k_s123", "context_length": 4096,
             "spec_steps": 4, "seed": 123},
            {"group": "M4", "level_id": "TARGET_ctx4k_pg19_phaseD",
             "run_id": "TARGET_ctx4k_pg19_phaseD_s42", "context_length": 4096,
             "spec_steps": 0, "seed": 42},
        ])
        out = ana.add_series(df)
        assert out["series"].nunique() == 2

    def test_coverage_flags_a_group_with_too_few_seeds(self, ana, tmp_path):
        df = ana.add_series(pd.DataFrame([
            {"group": "G", "level_id": "L", "run_id": "L_s123",
             "context_length": 4096, "spec_steps": 4, "seed": 123},
            {"group": "G", "level_id": "L", "run_id": "L_s456",
             "context_length": 4096, "spec_steps": 4, "seed": 456},
        ]))
        cov = ana.check_seed_coverage(df, out_dir=tmp_path)
        # A multi-seed group with n=2 is exactly the failure this guards.
        assert cov[cov["n_seeds"] > 1]["n_seeds"].max() == 2

    def test_coverage_passes_at_three(self, ana, tmp_path):
        df = ana.add_series(pd.DataFrame([
            {"group": "G", "level_id": "L", "run_id": f"L_s{s}",
             "context_length": 4096, "spec_steps": 4, "seed": s}
            for s in (42, 123, 456)
        ]))
        cov = ana.check_seed_coverage(df, out_dir=tmp_path)
        assert cov[cov["n_seeds"] > 1]["n_seeds"].tolist() == [3]


class TestA4Verifier:
    def test_absent_trace_is_unverifiable_not_a_pass(self, tmp_path):
        # "Could not check" must never read as "checked out".
        from src.analysis.acceptance import verify_csv_acceptance
        csv = tmp_path / "r.csv"
        pd.DataFrame([{"run_id": "nope_s42", "acceptance_rate": 0.5}]).to_csv(
            csv, index=False)
        (tmp_path / "per_token").mkdir()
        out = verify_csv_acceptance(csv, tmp_path / "per_token")
        row = out[out["run_id"] == "nope_s42"].iloc[0]
        assert row["verdict"] == "unverifiable_no_trace"
        assert row["ok"] is None or (isinstance(row["ok"], float)
                                     and np.isnan(row["ok"]))

    def test_verdict_column_distinguishes_all_states(self):
        from src.analysis.acceptance import verify_csv_acceptance
        import inspect
        src = inspect.getsource(verify_csv_acceptance)
        for v in ("match", "mismatch", "unverifiable_no_trace",
                  "unverifiable_no_csv_row"):
            assert v in src, v

    def test_matching_is_exact_run_id_only(self):
        from src.analysis import acceptance as acc
        import inspect
        src = inspect.getsource(acc.verify_csv_acceptance)
        # A normalized/prefix fallback would pair a CSV row with a different
        # run's trace, which is the failure this guards against.
        assert "run_id not in traces" in src
        assert "startswith" not in src.split("def verify_csv_acceptance")[1][:4000]


class TestD1Paired:
    @staticmethod
    def _write_pair(root: Path, a2: list[float], a3: list[float]) -> Path:
        """Write arm2/arm3 traces with matching seeds."""
        td = root / "per_token"
        td.mkdir(parents=True, exist_ok=True)
        import json
        for arm, vals in (("ARM2_llama3_native_128k_cap4k", a2),
                          ("ARM3_llama3_native_128k_nativedraft", a3)):
            for seed, alpha in zip((42, 123, 456), vals):
                gamma = 4
                n_acc = round(alpha * gamma)
                rows = [{"n_acc": n_acc, "spec_steps": gamma}
                        for _ in range(30)]
                with (td / f"{arm}_s{seed}.jsonl").open("w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
        return td

    def test_null_result_never_claims_equivalence(self, rep, tmp_path):
        # Identical arms: the CI spans zero, and the honest reading is "no
        # difference detected at this n" — NOT "the arms are equivalent",
        # which n=3 cannot support.
        td = self._write_pair(tmp_path, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        df = rep.d1_paired_arm2_arm3(td)
        summary = df[df["comparison"] == "arm2_minus_arm3"].iloc[0]
        assert "no difference detected" in summary["read"]
        assert "NOT evidence of equivalence" in summary["read"]
        # The bare claim must not appear on its own.
        assert summary["read"].strip() != "equivalent"

    def test_diff_is_paired_on_seed(self, rep, tmp_path):
        # Arms sharing no seed must yield nothing rather than an unpaired
        # comparison dominated by prompt variance.
        td = self._write_pair(tmp_path, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        df = rep.d1_paired_arm2_arm3(td)
        summary = df[df["comparison"] == "arm2_minus_arm3"].iloc[0]
        assert summary["n_seeds"] == 3
        assert summary["mean_diff"] == pytest.approx(0.25, abs=1e-6)
