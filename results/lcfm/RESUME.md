# RESUME: LCFM camera-ready finalization

Paused on 2026-10-13 at the user's instruction, at a safe point. Part 6 (branch
cleanup, merge, tags) has NOT started: no merge is open, no branch was deleted, no
tag was created.

Branch: `lcfm-camera-ready`. Commit: `416cb71` (WIP: paused before trimming the
camera-ready to 8 pages), pushed to origin with an explicit refspec.

## Where the run got to

| part | state | detail |
|---|---|---|
| 1 template and structure | IN PROGRESS | `manuscript/lcfm/{main.tex, neurips_2026.sty, references.bib}` exists and compiles. The style is the reviewed submission's own official NeurIPS 2026 package (from `manuscript/workshop/`), used with the same options plus `[final]`. `manuscript/arxiv/` is untouched. Main text currently ends on page 9 and the References heading is on page 10: **one page must still come out.** |
| 2 the six decisions | DONE in the text | D1 new title, D2 ceiling test confined to one labelled sentence in the system section plus Appendix E, D3 profiler table and figure cut, D4 released degeneracy numbers only (9/10, 1/10, 1/83 with the f16 row named) and the 9/15 census once in Appendix A marked "sidecars not released", D5 vLLM both directions with the mismatch in the same sentence, D6 Llama-2 perplexity and RULER appendices cut. Authors: the reviewed submission's own camera-ready block (Aman Kesarwani, Independent Researcher, amankesarwani@berkeley.edu). |
| 3 citations | DONE except one nice-to-have | Meta context parallelism replaced with the verified `yang2024contextparallelism` (arXiv:2411.01783, nine authors, from the arXiv API); vLLM DCP replaced with the verified `lee2026dcp` (title, four authors, 7 Aug 2026, URL, from the blog page); the NVIDIA co-design entry REMOVED because it could not be fetched, and the sentence that cited it rephrased. Every other cited entry has author, title, year and venue. Nice-to-have: five arXiv-based entries were being re-verified against the arXiv API when it returned HTTP 429; the retry is optional. |
| 4 em dashes | MOSTLY DONE | `manuscript/lcfm/main.tex` and `references.bib` are written with no `---` and no U+2014. `tables/ablation_summary.tex` had one and it is removed. Still to do: normalise the `# ---` comment separators in `scripts/lcfm_figures.py` (comments only, no drawn text contains an em dash) and run both gates. |
| 5 consistency pass | NOT STARTED | 5a number audit against the kit and the figure script output, 5b the same figures in abstract/intro/results/conclusion, 5c the reviewer-comment mapping table (reviews are in the repo at `manuscript/arxiv/REVIEW_ROADMAP.md`), 5d clean compile gate, 5e `results/lcfm/ORGANIZER_NOTE.txt`. |
| 6 branch cleanup and tag | NOT STARTED | Nothing was merged, deleted or tagged. |
| 7 FINAL_REPORT.txt | NOT STARTED | |

## Known problems to fix first

1. **Main text is one page too long.** Measured, not estimated: main text ends on
   page 9, References heading on page 10, total 14 pages. The LCFM limit is 8 pages
   of main text.
2. **One overfull box above the gate:** 16.81pt, from
   `\input{../../tables/ablation_summary.tex}` in Appendix D. The many 1.85pt
   overfull lines come from the large Appendix A table and are inside the 5pt gate.
3. **Affiliation discrepancy to confirm:** the reviewed submission's own block says
   "Independent Researcher"; the arXiv long version says "University of California,
   Berkeley (alumnus)". The camera-ready uses the reviewed submission's block.

## Exact next step

Trim the main text by about one page, recompile, and re-measure:

```
cd manuscript/lcfm && tectonic -X compile main.tex --keep-logs
pdftotext -layout main.pdf /tmp/lcfm.txt   # then find the page with the References heading
```

Then fix the Appendix D overfull box:

```
{\footnotesize\setlength{\tabcolsep}{4pt}\input{../../tables/ablation_summary.tex}}
```

Suggested cuts, in order of least damage: shorten the abstract (currently about 250
words and it repeats the introduction), compress the five introduction finding
paragraphs by pointing at the results subsections for the detail, tighten the
limitations list, and shrink the four figures slightly (currently 1.0, 1.0, 1.0 and
0.72 of the text width).

After that: Part 4 gates, then Part 5 (including ORGANIZER_NOTE.txt), then Part 6
(branch cleanup) only once the PDF is final, then Part 7.

## Environment state at pause

0 cloud instances (Lambda API), no dead-man timers and no capacity pollers running,
working tree clean apart from this file's commit. Budget used this run: $0, no GPU.
