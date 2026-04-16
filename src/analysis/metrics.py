"""Load and aggregate RASD ablation CSVs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Rows with tokens_generated below this are deterministic early-EOS outliers
# (confirmed by re-running with identical seed). See analysis/error_analysis.md.
SHORT_RUN_THRESHOLD = 20

# M3 ablation groups in paper-presentation order.
GROUPS = ["A1", "A2", "A3", "A4", "A5"]

GROUP_LABELS = {
    "A1": "Draft model",
    "A2": "Spec steps (k)",
    "A3": "KV block size",
    "A4": "Prefetch depth",
    "A5": "Target model",
}

# Per-axis ordering so plots + tables read naturally.
LEVEL_ORDER = {
    "A1": ["A1_tinyllama_1b", "A1_sheared_1b"],
    "A2": ["A2_k2", "A2_k4", "A2_k6", "A2_k8", "A2_k12"],
    "A3": ["A3_block256", "A3_block512", "A3_block1024", "A3_block2048"],
    "A4": ["A4_sync", "A4_async1", "A4_async2"],
    "A5": ["A5_llama2_7b", "A5_llama2_13b"],
}

# Display labels for plot x-axis / table first column.
LEVEL_LABELS = {
    "A1_tinyllama_1b":  "TinyLlama-1.1B",
    "A1_sheared_1b":    "Sheared-1.3B",
    "A2_k2":            "k=2",
    "A2_k4":            "k=4",
    "A2_k6":            "k=6",
    "A2_k8":            "k=8",
    "A2_k12":           "k=12",
    "A3_block256":      "256",
    "A3_block512":      "512",
    "A3_block1024":     "1024",
    "A3_block2048":     "2048",
    "A4_sync":          "sync",
    "A4_async1":        "async-1",
    "A4_async2":        "async-2",
    "A5_llama2_7b":     "Llama-2-7B",
    "A5_llama2_13b":    "Llama-2-13B",
}


def load_ablations(path: str | Path = "results/ablations/ablations.csv") -> pd.DataFrame:
    """Read the ablation CSV and coerce numeric columns."""
    df = pd.read_csv(path)
    numeric = ["tokens_generated", "time_sec", "throughput_tps",
               "acceptance_rate", "mean_latency_ms", "gpu_peak_mem_mb", "n_rounds"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_valid(df: pd.DataFrame, drop_short: bool = True) -> pd.DataFrame:
    """Keep only rows with status=ok and (optionally) tokens_generated >= threshold."""
    out = df[df["status"] == "ok"].copy()
    if drop_short:
        out = out[out["tokens_generated"] >= SHORT_RUN_THRESHOLD]
    return out


def per_level_agg(df: pd.DataFrame, metric: str,
                  groups: Iterable[str] = GROUPS) -> pd.DataFrame:
    """Per-level mean/std/n for a metric. One row per (group, level_id)."""
    rows = []
    for g in groups:
        sub = df[df["group"] == g]
        for level_id in LEVEL_ORDER[g]:
            vals = sub[sub["level_id"] == level_id][metric].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "group":   g,
                "level_id": level_id,
                "label":   LEVEL_LABELS[level_id],
                "n":       len(vals),
                "mean":    float(vals.mean()),
                "std":     float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min":     float(vals.min()),
                "max":     float(vals.max()),
            })
    return pd.DataFrame(rows)
