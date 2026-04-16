# M3 Error Analysis — short-run rows and filtering rationale

Written 2026-04-16. Source data: [results/ablations/ablations.csv](../results/ablations/ablations.csv)
(49 rows, all `status=ok`). All numbers reproducible via
`python scripts/compute_ablation_cis.py`.

## Why this file exists

All 49 ablation rows completed with `status=ok`, but four produced very short
generations (`tokens_generated < 20`). Before using the CSV for bootstrap CIs
and paper figures, we classify each short run as either a pipeline failure
(which would force a re-run) or a deterministic consequence of the sampled
prompt + config (which is legitimate and should be excluded from aggregate
statistics, not re-run). All four are deterministic; no re-runs are needed.

## The four short-run rows

| run_id | tokens | accept | tps | classification |
|---|---:|---:|---:|---|
| `A2_k2_s42`  | 6  | 0.000 | 1.28 | k=2 under-speculation + early EOS |
| `A2_k8_s42`  | 3  | 0.000 | 0.62 | k=8 over-speculation + early EOS  |
| `A1_tinyllama_1b_s456` | 9  | 0.036 | 1.53 | seed-456 short-prompt pattern     |
| `A2_k6_s456` | 14 | 0.030 | 1.65 | seed-456 short-prompt pattern     |

## Pattern 1 — Extreme `k` + seed-42 → deterministic early EOS

At `k=2` the draft model generates only 2 tokens/round; with high-entropy
Llama-2 outputs this is too few proposals to ever land an accepted continuation.
Every round falls back to target-only sampling, which for seed-42's prompt
happens to sample `</s>` within the first ~3 rounds. Same mechanism at `k=8`:
with 8 draft tokens the joint acceptance probability collapses (~0.5⁸ even
before any divergence), so all drafts are rejected and we again fall back to
target-only sampling that hits `</s>` early.

- **Not a bug** — confirmed by re-running with identical seed; produces the
  same 6 / 3 tokens byte-for-byte.
- **Not a k-sweep failure either** — `A2_k2_s123 = 256 tokens` and
  `A2_k8_s123 = 191 tokens` on the *same* config with a different seed. The
  combination of aggressive k and seed-42's prompt is what triggers early EOS.

## Pattern 2 — Seed 456 samples a naturally short prompt

Every seed-456 row clusters at 23–28 tokens generated (16/17 rows). The two
outliers `A1_tinyllama_1b_s456=9` and `A2_k6_s456=14` are just the shorter tail
of the same distribution — the prompt sampled at seed 456 reaches `</s>` in
fewer draft-verify rounds.

| seed | median `tokens_generated` | range |
|---|---:|---|
| 42  | ~120 | 3 – 256 |
| 123 | ~200 | 111 – 256 |
| 456 | ~26  | 9 – 120 |

Seed 456 isn't "broken" — the benchmark prompt distribution includes short
documents and seed 456 landed on one. Filtering by `tokens_generated >= 20`
drops these as low-information rows without re-running.

## Decision: filter threshold `tokens_generated >= 20`

For all bootstrap CIs and paper figures we drop rows below this threshold
(4 out of 49). This is implemented in
[src/analysis/metrics.py](../src/analysis/metrics.py) `SHORT_RUN_THRESHOLD = 20`
and applied by `filter_valid(df)`. The unfiltered CSV remains the source of
truth on disk.

**Effect on CIs:** `A1_tinyllama_1b` loses its seed-456 row, leaving n=2.
`A2_k2` and `A2_k8` each lose their seed-42 row, leaving n=2 each. All other
levels retain n=3.

## Determinism fingerprints in the filtered data

Confirm what the design predicts: ring block size (A3) and prefetch depth (A4)
should affect *timing only*, not the generated token stream. The data bears
this out.

- `A3_block{1024,2048}` and all three A4 levels produce **identical**
  `tokens_generated`, `acceptance_rate`, and `n_rounds` per seed
  (s42=148, s123=111, s456=28). Eight rows carry exactly the same compute
  fingerprint; only `throughput_tps` differs.
- `A3_block{256,512}` differ only because a smaller block reaches the EOS
  stopping point at a different round boundary (acceptance within CI band).

If a future M4 edit silently changes one of these fingerprints, the
[scripts/replay_m3_smoke.sh](../scripts/replay_m3_smoke.sh) guard will catch
it.

## Finding that should flag in the paper

**A1 (draft model) winner is not statistically clean.** After filtering,
TinyLlama-1.1B (n=2) has acceptance 0.066 [0.065, 0.068] and Sheared-LLaMA-1.3B
(n=3) has 0.059 [0.040, 0.072]. CIs overlap almost entirely. The earlier
README claim that "Sheared-LLaMA-1.3B small edge over TinyLlama-1.1B" is not
supported by the bootstrap — it appears to come from the pre-filter raw mean
which was distorted by the 9-token TinyLlama row on seed 456.

Action: soften the A1 finding in the paper to "within-CI tie" unless M4 runs
add seeds. A1 is a secondary axis; the primary story (k=4, block=1024) is
unaffected.
