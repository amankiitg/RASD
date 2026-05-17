# RASD — Ring Attention with Speculative Decoding

A reference implementation of **RASD**, an inference system that
integrates ring-attention sequence parallelism, NF4 KV-cache
quantization with chunked updates and a bf16 attention-sink prefix,
YaRN position scaling, and speculative verification into a single
end-to-end loop.

On 8×A100 80 GB SXM4, RASD runs Llama-2-7B inference at
**1M-token context with 40 GB peak per rank** — a regime where the
standard HuggingFace FlashAttention-2 inference path runs out of
memory at 128k single-rank.

**Paper:** [`manuscript/arxiv/main.pdf`](manuscript/arxiv/main.pdf)
(arXiv ID pending). **License:** [MIT](LICENSE).

---

## Headline numbers

| Claim | Number |
|---|---|
| Max context reached on 8×A100 80 GB | **1,048,576 tokens (1 M)** |
| Per-rank peak memory at 1 M | **39.3 GiB** |
| Vanilla HF FA-2 `generate()` ceiling, single-rank | **32 k** (OOMs at 128 k) |
| Throughput speedup vs target-only baseline, 128 k (3-seed mean) | **1.26×** |
| Throughput speedup vs target-only baseline, 256 k (3-seed mean) | **1.76×** |
| Communication cost on speculator rank | **≤ 1.2 %** of wall |
| Per-round acceptance distribution at long context | **bimodal** (≈50 % α=0, ≈50 % α∈[0.25, 0.5]) |

The full matrix and the M3 ablation sweeps that derived each design
knob are reported in the paper.

---

## Quick start

```bash
git clone https://github.com/amankiitg/RASD.git
cd RASD

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-lock.txt

# flash-attn must be installed separately (uses --no-build-isolation):
pip install --no-build-isolation "flash-attn>=2.4.0,<3"

# Editable install of the RASD package
pip install -e .

# Sanity checks
python check_env.py
pytest tests/ -q --ignore=tests/test_long_context_smoke.py
```

The full set of pinned versions is in
[`requirements-lock.txt`](requirements-lock.txt). Three immutable
git tags point to the exact code that produced each milestone:

* `m3-complete` — the M3 ablation grid (Figure 2 source).
* `m4-phase-c-complete` — the headline matrix (throughput, baseline,
  PPL, profiler, ceiling tests).
* `m4-phase-d-complete` — per-position acceptance traces, saved
  generations, and the short-context PG-19 control.

---

## Reproducing the paper figures and tables

Every figure and every LaTeX table in the manuscript regenerates
**from committed artifacts**, with no GPU required:

```bash
python scripts/aggregate_final_results.py     # results/final/final_results.json
python scripts/plot_figure1.py --show-hf      # F1 — throughput vs context
python scripts/plot_figure3.py                # F3 — time-breakdown bars
python scripts/plot_figure4.py                # F4 — α vs round + bimodality
python scripts/emit_figure5_qualitative.py    # F5 — qualitative comparison
python scripts/emit_main_tables.py            # tables/main_*.tex
python scripts/error_analysis_alpha.py        # analysis/error_analysis.md (F8)
```

To recompile the PDF after editing the source:

```bash
cd manuscript/arxiv && tectonic main.tex
```

(Install [tectonic](https://tectonic-typesetting.github.io/) via
`brew install tectonic` or your distribution's package manager.)

### Reproducing the GPU experiments

Re-running the full Phase C matrix on a fresh 8×A100 80 GB SXM4 pod
takes ~1.5 hours end-to-end. The bootstrap is one command:

```bash
WANDB_API_KEY=... HF_TOKEN=... HF_HOME=/home/ubuntu/hf_cache \
  bash scripts/phase_c_pod_session.sh
```

See [`REPRODUCE.md`](REPRODUCE.md) for the operator walkthrough and
[`docs/dev/PHASE_C_RUNBOOK.md`](docs/dev/PHASE_C_RUNBOOK.md) for the
detailed runbook.

---

## Repository layout

```
LICENSE                    MIT.
README.md                  This file.
REPRODUCE.md               Operator-grade reproduction walkthrough.
requirements-lock.txt      Pinned package versions used in the paper.
pyproject.toml             Editable install descriptor.

src/                       The RASD inference engine.
├── models/                Engine, ring-attention layers, NF4 cache,
│                          speculative verify loop, YaRN scaling.
├── baselines/             Reference ring-attention + sliding-window forwards.
├── analysis/              Metrics: throughput, acceptance, profiler aggregation.
└── utils/                 Device selection, helpers.

scripts/                   CLI entry points.
├── phase_c_pod_session.sh           Phase C orchestration on a fresh pod.
├── phase_d_rerun_session.sh         Per-token + qualitative re-run.
├── aggregate_final_results.py       Build results/final/final_results.json.
├── plot_figure{1,3,4}.py            Figure regeneration.
├── emit_figure5_qualitative.py      F5 qualitative table.
├── emit_main_tables.py              LaTeX tables for the manuscript.
├── error_analysis_alpha.py          F8 bimodality analysis.
├── score_ruler_niah.py              RULER niah post-hoc scorer (future-work infra).
└── (others) preprocess_pg19.py, benchmark_baselines.py, ...

configs/                   YAML experiment specs.
data/                      Memmap caches (gitignored; populated at run time).
results/                   Committed CSVs, sidecars, profiler JSONs, generated text.
figures/                   Vector PDFs + PNGs of every paper figure.
tables/                    LaTeX-input-ready table fragments.
analysis/                  Analysis reports (F8 error analysis, etc.).
manuscript/                arXiv preprint LaTeX source + bibliography.
literature_review/         Mentor roadmap, lit-review memo, paper-tracking sheet.
tests/                     470-test pytest suite.
docs/                      Public-facing docs.
docs/dev/                  Working-tree planning docs (M3/M4/M5 plans, runbooks).
```

---

## What this codebase does *not* claim

* **It is not a long-context capability benchmark.** The paper
  reports systems metrics (throughput, memory, acceptance). The
  long-context-capability evaluation (LongBench, RULER, ∞Bench) is
  explicitly out of scope; we ship the RULER niah-scoring
  infrastructure so follow-up work can build on it directly.
* **It does not claim a modeling contribution.** Llama-2-7B is a
  fixed substrate; substituting a long-context-trained base model
  (Llama-3.1-128k, Qwen-2.5-1M) is straightforward and is documented
  as the highest-leverage future-work direction.
* **It is single-instance only.** All measurements are on 8×A100
  80 GB SXM4 on one node. Tensor parallelism (TP > 1) and multi-node
  ring attention are future work.

---

## Acknowledgments

This work was done as Aman Kesarwani's research project, advised by
Dr. Raj Dandekar. Compute was provided by Lambda Cloud research
credits. Implementation builds on PyTorch, the HuggingFace
`transformers` library, `bitsandbytes` (NF4), and `flash-attn`
(FlashAttention-2).

## Citation

Citation BibTeX will be added once the arXiv ID is assigned. For
now, please cite the GitHub repository.

```bibtex
@misc{kesarwani2026rasd,
  author = {Aman Kesarwani and Raj Dandekar},
  title  = {{RASD}: Ring Attention with Speculative Decoding for
            Million-Token Language Model Inference},
  year   = {2026},
  url    = {https://github.com/amankiitg/RASD},
  note   = {arXiv ID pending}
}
```

## Companion documents

* [`REPRODUCE.md`](REPRODUCE.md) — operator-grade reproduction walkthrough.
* [`docs/dev/README.md`](docs/dev/README.md) — index of dev-internal
  planning docs (M3 ring-integration plan, M4 1M-context plan, M5
  manuscript plan, Phase C runbook, publication strategy). These are
  the working-tree chronicles kept for full reproducibility audit.
* [`docs/M3_mentor_summary.md`](docs/M3_mentor_summary.md) — mentor
  summary at the end of M3 (kept for context).
