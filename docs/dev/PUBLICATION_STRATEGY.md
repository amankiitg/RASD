# RASD — Publication Strategy

Authored 2026-05-06. Tracks venue targets, deadlines, and execution timeline
for publishing M3 and M3+M4 results.

## Venue priority

```
PRIMARY:  MLSys 2027   (deadline ~Oct 30, 2026 | conf May 2027)
BACKUP:   NeurIPS 2027 (deadline ~May 22, 2027 | conf Dec 2027)
```

Plus low-stakes parallel-safe outlets:

```
arXiv preprint               : as soon as M3 R6.5 lands (target June 2026)
NeurIPS 2026 workshop (ENLSP /
ML-for-Systems)              : deadline ~Aug-Sept 2026
```

## Why MLSys primary, not NeurIPS

- MLSys is the right venue fit. Composition + careful ablation + engineering
  insights are paper-defining contributions there, not "engineering, not
  research" complaints. FlashAttention (Tri Dao 2022) was an MLSys paper —
  same lane.
- NeurIPS main track skews methods-heavy. Reviewers there often reject
  systems papers for "no algorithmic novelty." 64k context is no longer a
  headline number (Claude/Gemini are public at 200k+).
- For frontier-AI-lab hiring (the career goal here), an MLSys paper +
  arXiv preprint + open-source release is a stronger signal for
  inference-infra / applied research roles than a single ICLR/NeurIPS
  paper would be. The work and the community recognition are what
  matter, not venue ranking.
- MLSys deadline (Oct 30) is only one month later than ICLR (Sept). The
  prestige delta does not justify the harder reviewer lottery for
  systems-content work.

## Why NeurIPS 2027 backup, not MLSys 2028

- If MLSys 2027 rejects (decision late Jan 2027), NeurIPS 2027 main track
  (May 22, 2027 deadline) is the next clean shot.
- By then, M4's full 1M-context results should be in: "first 1M-context
  speculative decoding with ring + TP + spec composition" is a NeurIPS-grade
  headline that M3 alone is not.
- Sequential ICLR/ICML/NeurIPS chasing across 9-12 months is bad strategy
  in a fast-moving field. The space is moving too quickly to sit on results
  that long.

## Why skip NeurIPS 2026 main track

**REVISED 2026-05-06 (PM):** The NeurIPS 2026 main-track paper deadline
is **today, May 6, 2026 AOE** ([source](https://neurips.cc/Conferences/2026/Dates))
— not May 22 as originally noted. The Evaluations & Datasets track has
the same deadline. Both gates close at AOE end-of-day = ~30 hours from
mid-day UTC May 6.

This is operationally infeasible:

- Phase C (the bundled pod session) is ~10 hours by itself
- Lambda 8× A100 capacity is currently zero across all regions
- M3+M4 paper draft does not yet exist; experiments.md only covers M3
- Even if everything went perfectly we cannot deliver M3+M4 with
  Fig 1/3/4/5, error-analysis, ablation tables, and a polished
  manuscript in 30 hours

A rushed submission with un-validated 1M-context results is worse
than no submission; bad reviews go on file even after rejection.

**Substitute (still feasible):** NeurIPS 2026 workshops. Long-context /
efficient-inference workshops typically have late-July through
mid-September deadlines. ENLSP, ML-for-Systems, and similar venues
are the intended audience for this work. Non-archival, much higher
accept rate, still on CV.

## Why skip ICLR / ICML

- ICLR 2027 deadline (~Sept 2026) is only 1 month before MLSys 2027 — same
  window, weaker fit.
- ICML 2027 deadline (~Jan 2027) is too late; sitting on results 8+ months.
- Big-Three sequential strategy (ICLR → ICML → NeurIPS) is appropriate for
  slow-moving theory work. Not this.

## Execution timeline

| Month | Goal | Output |
|---|---|---|
| **May 2026** (now) | Land R6.5 cleanly. Lambda capacity polling. Scoped ablation (A1×A2×A5). | R6.5 results: α curves at 64k×W=8 |
| **June 2026** | M3 writeup. arXiv preprint. GitHub code release. Twitter/blog post. | arXiv 2606.xxxxx + RASD repo public |
| **June-July 2026** | Start M4: tensor parallelism integration. | TP wrappers + 2D process group + tests |
| **August 2026** | M4 continues: draft ring attention + 1M RoPE strategy (NTK/YaRN). | Working 256k→512k smoke |
| **August 2026** | NeurIPS 2026 workshop submission (M3 + preliminary M4). | Workshop submission ~Aug 25 |
| **September 2026** | M4 1M-context smoke on Lambda. Iterate. | 1M-context α + memory profile |
| **October 2026** | Paper writing, baselines, ablation surfaces, related work. Repo cleanup for reproducibility. | MLSys 2027 draft |
| **Oct 30, 2026** | **MLSys 2027 submission deadline.** | Submission ID |
| **Nov-Dec 2026** | Workshop presentation at NeurIPS 2026 (Dec). Continue extensions. | Talk + community feedback |
| **Late Jan 2027** | MLSys 2027 decision. | Accept/reject |
| **Feb-Mar 2027** | If accepted: camera-ready prep, May talk prep. If rejected: revise based on reviews for NeurIPS. | Revised paper |
| **May 22, 2027** | If MLSys rejected: **NeurIPS 2027 submission**. | Backup shot |
| **May 2027** | If MLSys accepted: present in person. | Conference talk |

## Three blockers to flag now

1. **Lambda capacity for R6.5** — without this, June arXiv slips, August
   workshop slips, October MLSys slips. Background polling for 8x A100 80GB
   SXM4 needs to be running. Fallback: 40 GB SXM2 at scoped-down ctx if
   80 GB stays unavailable. (Note: per
   `project_compute_provider_choice.md`, do NOT switch to RunPod — Lambda
   credit must be spent.)
2. **TP integration for M4** — biggest engineering risk. Estimated 2 weeks
   nominal, 4-6 weeks if NCCL bandwidth contention surprises. Buffer August
   for this.
3. **Reproducibility burden** — MLSys reviewers expect runnable code,
   replicable results, clear ablation tables. Allocate ~1 week dedicated
   effort in October to clean up the repo before submission.

## What to NOT do (policy reminders)

- **No simultaneous submissions to peer-reviewed venues.** All Big Three
  + MLSys forbid dual-submission. Sequential only.
- Workshops are exempt — non-archival, allowed in parallel with main-track
  submission elsewhere.
- arXiv is allowed concurrently with all venues.
- Don't submit to NeurIPS / MLSys before R6.5 results are clean.
- Don't dual-submit to MLSys + NeurIPS main simultaneously.
- Don't wait past November to start M4. The 1M results need to be in the
  MLSys submission to make the contribution complete.

## Comparison framing for related work

For positioning vs other published work at submission time:

- **FlashAttention-2/3** (Tri Dao, MLSys 2022/2023) — published, kernel-level.
  RASD composes ring (sequence-parallel) on top of FA-2; not in conflict.
- **Speculative Decoding originals** (Leviathan et al. 2023, Chen et al. 2023)
  — single-rank baseline that RASD extends to multi-rank long-context.
- **EAGLE / Medusa** — orthogonal speculative decoding methods (different
  draft architecture). RASD's contribution is the systems composition, not
  the speculation algorithm.
- **Ring Attention** (Liu et al. 2023, Korthikanti SeqPar) — sequence-
  parallel attention, no spec decoding. RASD adds spec layer.
- **Gap RASD fills:** rigorous composition + ablation of all of these for
  long-context inference, with new engineering findings (decorative-prefetcher
  bug, dual-cache decode, RoPE-stretch degradation in small drafts,
  TP+SP+spec scheduling).

## Companion docs

- [M3_RING_INTEGRATION_PLAN.md](M3_RING_INTEGRATION_PLAN.md) — current
  ring/spec integration status, R6 issues
- [M4_PLAN.md](M4_PLAN.md) — M4 work items (TP, draft ring, 1M context)
- Memory: `project_publication_strategy.md` — short pointer to this file
- Memory: `project_compute_provider_choice.md` — Lambda over RunPod rationale

## Revision log

- 2026-05-06 (PM): User asked "can we do NeurIPS this year?" — answer
  no for NeurIPS 2026 main + Evaluations & Datasets tracks (full paper
  deadline is **today, May 6, 2026 AOE** per neurips.cc/Conferences/2026/Dates;
  M3+M4 paper not draftable in 30 hours). Still yes for NeurIPS 2026
  workshops (later in summer) + arXiv preprint + MLSys 2027 primary.
  Updated "Why skip NeurIPS 2026 main track" section with corrected
  deadline (was wrongly noted as May 22 in the original draft). Plan
  unchanged: MLSys 2027 primary, NeurIPS 2026 workshop parallel,
  arXiv preprint when M4 paper polished.
- 2026-05-06 (AM): Created. Initial plan after extended venue discussion.
  Considered ICLR 2027 → demoted to "skip" because MLSys is the better
  fit and the 1-month window difference doesn't justify the worse
  reviewer match. Considered NeurIPS 2026 main → demoted to "skip"
  (deadline math was wrong then; corrected this PM).
