"""The scaling figure must not draw a rounded number or invent a clean row.

Two defects were found in `scripts/mlsys_plot_scaling.py` while regenerating the
figure for the sampling session, both of the class the campaign keeps paying for:

1. `throughput_tps` (L5). The CSV field carries two significant figures — the
   sampling session's 1M row is 0.12 where `tokens_generated/time_sec` is 0.12687 —
   and the plot was drawing the field. Every mean in the figure was therefore
   rounded, at up to 5% on the smallest rows.
2. A row with no token sidecar was plotted as NON-DEGENERATE (L13, the silent-drop
   class). `loop` was computed as `bool(ids) and bool(row_reasons(st))`, so a missing
   sidecar made `loop` False, and `no_ids` was recorded but never used: a row that
   might be a greedy loop went into the means as a clean row. Which is the one thing
   this figure exists to prevent.

The test asserts on outputs: the value loaded, and the CLI's own report for a
directory that contains one measurable and one unmeasurable row.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOT = REPO / "scripts" / "mlsys_plot_scaling.py"


def _load():
    spec = importlib.util.spec_from_file_location("_plot_scaling_under_test", PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEADER = ("run_id,context_length,status,time_sec,throughput_tps,acceptance_rate,"
          "spec_steps,tokens_generated,max_new_tokens,doc_id")


def _fixture(tmp_path: Path) -> Path:
    d = tmp_path / "session"
    (d / "tokens").mkdir(parents=True)
    with (d / "session_a_TEST.csv").open("w", newline="") as fh:
        fh.write(HEADER + "\n")
        # 128 tokens / 1009.0 s = 0.126857; the ROUNDED field says 0.12 (a 5% error).
        fh.write("MEASURED,1048576,ok,1009.0,0.12,0.5,4,128,128,pg19_train_0\n")
        fh.write("UNMEASURABLE,1048576,ok,1000.0,0.13,0.5,4,128,128,pg19_train_1\n")
    json.dump({"generated_token_ids": list(range(200))},
              (d / "tokens" / "MEASURED.json").open("w"))
    # UNMEASURABLE deliberately gets no sidecar.
    return d / "session_a_TEST.csv"


def test_toku_per_second_is_exact_not_the_rounded_field(tmp_path):
    mod = _load()
    rows = mod.load(_fixture(tmp_path), tmp_path / "session" / "tokens")
    by = {r["run_id"]: r for r in rows}
    assert abs(by["MEASURED"]["tps"] - 128 / 1009.0) < 1e-12, \
        "the plot must use tokens_generated/time_sec, not throughput_tps"
    assert by["MEASURED"]["tps"] != 0.12, "the rounded field was used"


def test_a_row_with_no_token_sidecar_is_reported_and_excluded(tmp_path):
    """The CLI must say so and keep the row out, not plot it as clean."""
    d = _fixture(tmp_path)
    out = tmp_path / "scaling.png"
    proc = subprocess.run(
        [sys.executable, str(PLOT), str(d), "--tokens-dir", str(d.parent / "tokens"),
         "--out", str(out)],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "cannot be tested for repetition" in proc.stdout, \
        "an unmeasurable row must be announced"
    assert "UNMEASURABLE" in proc.stdout
    assert "spec rows plotted (non-degenerate): 1" in proc.stdout, \
        "only the measurable row may be plotted"
    assert out.exists()
