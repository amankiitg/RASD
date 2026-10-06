# results/mlsys — MLSys experiment program artefacts

Produced by the MLSys experiment program (branch `mlsys-experiments`).
Run everything with one command: `bash scripts/mlsys_pod_session.sh`.

## Inputs produced by the pod session

| File | Phase | Contents |
|------|-------|----------|
| `arm1_llama2_yarn.csv` | 1 | Llama-2-7B + YaRN @128k, Sheared-1.3B draft (reproduction control) + matched `spec_steps=0` baseline |
| `arm2_native_cappeddraft.csv` | 1 | Llama-3.1-8B @128k native target, Llama-3.2-1B draft capped at 4k + baseline |
| `arm3_native_nativedraft.csv` | 1 | Llama-3.1-8B @128k native target, draft at full native window + baseline |
| `pg19_multiseed.csv` | 2 | PG-19 dose-response 4k/8k/1M (seeds 123/456) and the 1M target-only baseline (seeds 42/123/456) |
| `llama2_matrix_multiseed.csv` | 2 | Llama-2 matrix 128k/256k/512k/1M for seeds 123/456 |
| `per_token/` | 2 | Per-round acceptance traces (`.jsonl`). Seed-42 traces are copied from `results/final/per_token/`; seeds 123/456 are new |
| `bf16_draft_isolation.csv` | 3 | 64k, draft in 4-bit FP4 weights (bitsandbytes) vs bf16, 3 seeds |
| `vllm_baseline.csv` | 5 | vLLM @128k, throughput in RASD's unit + `unit_matched` flag |
| `logs/`, `.markers/` | all | Per-stage logs and resume markers |
| `gpu_hours.csv` | all | Wall time, GPU-hours and node cost per stage |

## Derived artefacts

| File | Contents |
|------|----------|
| `MASTER_TABLE.txt` | Old vs new number for each reviewer / roadmap item. `(pending GPU run)` means the cell does not exist yet — it is never back-filled. |
| `summary_with_ci.csv` | Bootstrap CIs on every metric per (group, level) |
| `trace_summary.csv` | Per trace: `alpha_round`, `alpha_iid`, `iid_ks`, `p_zero` |
| `dip_test.csv` | Hartigan dip test, one row per trace |
| `dip_by_context.csv` | Dip aggregated across seeds, with `n_seeds` reported |
| `acceptance_accounting.csv` | Verification that the CSV `acceptance_rate` is the per-round alpha (AbH52 #1) |
| `acceptance_accounting_seed42_pg19.csv` | Same check run against the already-committed seed-42 PG-19 rows |

Regenerate the derived artefacts at any time:

```bash
python scripts/mlsys_analysis.py --results-dir results/mlsys --trace-dir results/mlsys/per_token
python scripts/mlsys_master_table.py
```

## Acceptance conventions (reviewer AbH52 #1)

`alpha_round = SUM(n_acc) / (n_rounds * gamma)` — the mean per-round
accepted-prefix fraction. **This is what `acceptance_rate` reports**, and
`acceptance_accounting.csv` verifies it against the traces.

`alpha_iid` is a *different, derived* quantity: the single per-token
probability a memoryless model would need to reproduce the same mean.
On the committed seed-42 traces it is 0.32–0.87 versus `alpha_round`
values of 0.12–0.71, with KS distances of 0.05–0.14 — i.e. the i.i.d.
assumption is decisively rejected. Do not substitute one for the other.

## Notes

- `MASTER_TABLE.txt` reports the seed count behind every number. A
  single-seed cell is not multi-seed evidence.
- `unit_matched=no` rows in `vllm_baseline.csv` must be excluded from any
  speedup ratio against RASD's `throughput_tps`.
