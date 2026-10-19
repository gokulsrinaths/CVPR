# Amendment A23 to pre-registration v1: second seed of the A-OKVQA lock-out and a larger model

Appended 2026-10-05, before any run below. v1 and A0-A22 unchanged. Motivation (a simulated review of the manuscript): the A-OKVQA lock-out at d = 0.99 rests on one model and one seed, and the second domain on one model family size.

## What has been seen (not predictions)
All of A21 and A22 (analysis/a21_results.md, a22_results.md): for Qwen2.5-VL-3B on A-OKVQA (seed 20261021) g(intact) = +33.9 / +24.5 / +8.1 / +2.6 / +0.35 points at d = 0.25 / 0.70 / 0.92 / 0.97 / 0.99, V* = 2 / 3 / 3 / 4 / 7; second seed (20261022) only at d = 0.25 / 0.70 / 0.92 (+33.3 / +23.5 / +8.0); InternVL3-2B +37.6 / +28.9 / +11.7. No Qwen2.5-VL-7B run on A-OKVQA and no second-seed run at d = 0.97 or 0.99 has been seen.

## Runs (6; A30 GPUs, A-OKVQA split of A21, length shortcut, recipe and arms of A21)
- **Part B (second seed):** Qwen2.5-VL-3B, seed 20261022, d in {0.97, 0.99} (2 runs, no gradient checkpointing). Together with the A21 run at d = 0.92 of the same seed they give a second 0.92/0.97/0.99 curve.
- **Part C (larger model):** Qwen2.5-VL-7B, seed 20261021, d in {0.25, 0.92, 0.99} and d = 0.70 (4 runs, gradient checkpointing). Scripts: `scripts/a21_run.py` (unchanged), analysis `scripts/a23_analysis.py`, hashed before the first run.
All six on A30 GPUs (the shared CUDA 12.6 environment), so the second-seed curve and the 7B curve use the hardware of A21.

## Rules
- **B1 (lock-out replicates, seed 2):** g(0.99) <= 2.0 points and V*(0.99) = 7.
- **B2 (monotone, seed 2):** g(0.92) > g(0.97) > g(0.99) (g from the A21 run at 0.92 of seed 20261022).
- **C1 (dose-response, 7B):** g(0.25) - g(0.92) >= 5.0 points and g(0.25) > g(0.70) > g(0.92).
- **C2 (boundary shift, 7B):** V*(0.92) >= V*(0.25) + 1.
- **C3 (lock-out at 0.99, 7B; reported):** g(0.99) <= 2.0 and V*(0.99) = 7. **Expectation, not a rule:** B1, B2, C1 hold; C2 and C3 may fail.
Shortest-option pick rate and accuracy on unplanted questions are reported, not tested.

## Reading rule (fixed)
- B1 and B2 hold: the paper states that the lock-out at d = 0.99 was reproduced in a second seed; if B1 fails, the lock-out claim is reduced to "g <= +2.6 points at 0.97 and near zero at 0.99 in one of two seeds" and the abstract drops the lock-out strength comparison.
- C1 holds: the dose-response is stated for two model sizes on A-OKVQA; C2, C3 are reported as they come out and the paper states for which model the lock-out was observed.
Results are reported whatever they are; no run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One benchmark, one shortcut; two seeds at most for 3B and one for 7B; the second-seed curve at 0.92 comes from A21.

## Amendments log
(A0-A23.)
