"""Analysis utilities for RASD results.

metrics   — CSV loader + per-level aggregation
bootstrap — percentile-bootstrap confidence intervals
figures   — matplotlib defaults + figure saver
tables    — LaTeX booktabs emitter

All modules read-only on results/*.csv; outputs land in figures/, tables/,
or results/final/.
"""
