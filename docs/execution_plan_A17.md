# Execution plan for A17 (shortcut dose-response and the visual veto boundary)

Written 2026-10-04 before any A17 run. Not a pre-registration (that is `preregistration_v1_amendment_A17.md`); it records the audit, the cost and the order.

## 1. Audit of what exists (v5; nothing is overwritten)
- Paper: `paper/main.tex`, `supp.tex` (v5 PDFs in `cvpr_paper/`); 15 amendments A0-A15 frozen and hashed in `analysis/preregistration_v1.sha256`; job registry with 71 rows; release bundle with 127 verified hashes.
- Reusable infrastructure: PMC-VQA split (`data/pmcvqa/split_a11`, article-disjoint, donors); the A11 recipe and readout (`scripts/pmc_run*.py`); the filler and planting code (`len_rand.py`, `len_implant.py`, A12/A13); image degradation arms (A15); analysis patterns (`pmc_summarize.py`, `a15_*`); the freezing procedure (write amendment, scp to `docs/`, `chmod a-w`, `sha256sum` appended to the chain, then submit); A30 execution via `sbatch` (H100 queue saturated; gradient checkpointing for memory).
- Existing results that bear on the question (development knowledge, not confirmatory): original PMC-VQA (cue 20-25%, gap +13.7), planted 92% (gap about 0, seed 20261003), blur at two radii (A15), position and intermediate doses: none.
## 2. What can be computed without new GPU runs
Only two dose points (no-filler original, 92%) and two blur levels exist, from a single seed per family: enough to motivate the design, not to test it. Everything confirmatory needs new runs. CPU-only parts: the aggregate analysis and the figures.
## 3. New GPU runs and cost (A30, 3B bf16 with gradient checkpointing; estimates from the A16 smoke runs and A15 evaluations)
| Stage | Runs | Per-run time (est.) | GPU-hours |
|---|---|---|---|
| Batch 1: length doses x 6, seed 20261011 (3 runs also with 5 downsampling arms) | 6 | 2.0-2.5 h | ~14 |
| Stop check (pre-registered): proceed only if g(0.25) - g(0.92) >= 5 points | - | - | - |
| Batch 2: length doses x 6, seeds 20261012-13 | 12 | 2.0 h | ~24 |
| Second shortcut (position) x 3 doses, seed 20261011 | 3 | 2.0 h | ~6 |
| A18 held-out prediction: InternVL3-2B x 3 doses (gc), Qwen-7B x 3 doses (gc), seed 20261011 | 6 | 3-4.5 h | ~24 |
| Optional hysteresis (separate pre-registration, only if H1 and H2 hold) | 2-3 | 2.5 h | ~7 |
Total confirmatory without hysteresis: 27 runs, about 68 GPU-hours; wall clock about 8-10 hours if 8 A30 GPUs are available in parallel (they were on 2026-10-03/04). A16 (7B seeds, OmniMedVQA; 7 jobs, about 30 GPU-hours) runs concurrently.
## 4. Order (cheapest valid first)
1. A16 jobs (running). 2. A17: freeze, smoke, batch 1; stop check. 3. Batch 2 and the second shortcut. 4. `dose_analysis.py`, figures, results report. 5. A18 (quantitative prediction from the Qwen-3B fit) before any InternVL3 or Qwen-7B dose run; run; test. 6. Hysteresis only if warranted. 7. Restructure the manuscript only as the A17 reading rule allows.
