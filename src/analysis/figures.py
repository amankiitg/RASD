"""Matplotlib defaults + figure saver for paper-ready output."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_rcparams():
    """Apply paper-ready matplotlib rcParams. Call once before plotting."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
    })


def save(fig, path: str | Path, formats: tuple[str, ...] = ("pdf", "png")):
    """Save a figure in multiple formats next to each other."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out = p.with_suffix(f".{ext}")
        fig.savefig(out)
    plt.close(fig)
