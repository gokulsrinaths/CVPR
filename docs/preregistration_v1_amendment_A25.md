# Amendment A25 to pre-registration v1: more seeds for the A-OKVQA dose-response and lock-out

Appended 2026-10-05, before any run below. v1 and A0-A24 unchanged. Motivation (two simulated reviews): on A-OKVQA, Qwen-7B and InternVL3-2B have one seed and Qwen-3B two; the lock-out at d = 0.99 rests on two seeds of Qwen-3B and one of Qwen-7B.

## What has been seen (not predictions)
All of A21-A24 (analysis/a21_results.md ... a24_results.md). For A-OKVQA, length shortcut: Qwen-3B seeds 20261021/22: g = +33.9/+33.3 (d = 0.25), +24.5/+23.5 (0.70), +8.1/+8.0 (0.92), +2.6/+3.3 (0.97), +0.35/+0.09 (0.99); Qwen-7B seed 20261021: +34.2, +22.8, +6.5, +0.09 at 0.25/0.70/0.92/0.99; InternVL3-2B seed 20261021: +37.6, +28.9, +11.7 at 0.25/0.70/0.92 (not run above 0.92). No run of A25 has been seen.

## Runs (11; A30 GPUs, split, recipe and arms of A21; gradient checkpointing for the 7B and InternVL3-2B runs)
- **Part S1 (Qwen2.5-VL-7B, seed 20261022):** d in {0.25, 0.70, 0.92, 0.99} (4 runs).
- **Part S2 (InternVL3-2B, seed 20261022):** d in {0.25, 0.70, 0.92, 0.99} (4 runs; first runs of this model at 0.99).
- **Part S3 (Qwen2.5-VL-3B, seed 20261023):** d in {0.92, 0.97, 0.99} (3 runs).
Scripts: `scripts/a21_run.py` (unchanged), analysis `scripts/a25_analysis.py`, hashed before the first run.

## Rules (each series separately; series = model and seed)
- **D1 (dose-response; S1, S2):** g(0.25) - g(0.92) >= 5.0 points and g(0.25) > g(0.70) > g(0.92).
- **D2 (boundary shift; S1, S2):** V*(0.92) >= V*(0.25) + 1.
- **L1 (lock-out at d = 0.99; S1, S2, S3):** g(0.99) <= 2.0 points and V*(0.99) = 7.
- **L2 (monotone; S3):** g(0.92) > g(0.97) > g(0.99).
**Expectation, not a rule:** all hold; InternVL3-2B, whose benefit at d = 0.92 (+11.7) is the largest seen, may not lock out at 0.99.

## Reading rule (fixed)
For every series that meets L1 the lock-out at d = 0.99 on A-OKVQA is stated as observed for that model and seed; the paper then reports it as observed in n of m series with n and m counted over all A21-A25 series at d = 0.99. A series that fails L1 is reported with its g(0.99) and the paper states which model deviates; the claim "the image is locked out at 0.99 on A-OKVQA" is then restricted to the models for which L1 holds. D1/D2 results are added to the dose-response seed counts. Results are reported whatever they are; no run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One benchmark, one (length) shortcut; at most three seeds per model; InternVL3-2B has no run at 0.97.

## Amendments log
(A0-A25.)
