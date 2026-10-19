# Amendment A22 to pre-registration v1: where lock-out begins on a general-domain benchmark, a second non-medical shortcut, and a second seed for the held-out models

Appended 2026-10-04, before any run below. v1 and A0-A21 unchanged (A21 runs are still in progress; its rules and analysis script are not modified).

## What has been seen (not predictions)
Four of the nine A21 runs had finished and were summarised before this text was written (Qwen2.5-VL-3B, A-OKVQA, length shortcut, seed 20261021 at d = 0.25 / 0.70 / 0.92; seed 20261022 at d = 0.25 only): image benefit g(intact) = 33.89 / 24.45 / 8.12 points (seed 20261022, d = 0.25: 33.28); blur-ladder boundary V* = 2 / 3 / 3 (7 = none); shortest-option pick rate 0.248 / 0.669 / 0.918; accuracy on the original unplanted E1 questions with the true image 82.8 / 82.5 / 82.6% (with black image 49.6 / 49.8 / 48.6%). So the benefit falls but does not reach zero at d = 0.92, and the accuracy on unplanted questions does not fall. Under the A21 rules R1 and R3 and the pick-rate part of R2 hold for this series, the accuracy-drop part of R2 and R4 fail; the A21 verdict will be computed by the frozen script after all nine runs. The InternVL3-2B A21 series, the remaining Qwen-3B runs and everything in A22 below have not been seen.

## Runs (11)
All on the RTX PRO 6000 Blackwell GPUs of the cluster (partition mb-r6000) with a separate Python environment (PyTorch 2.7.1 for CUDA 12.8, because the shared environment does not support this GPU; transformers 5.10.2 and peft 0.19.1 are unchanged). Hardware and CUDA version therefore differ from A17-A21 (A30, CUDA 12.6); splits, scripts, recipe, prompts and seeds' roles are unchanged, and results are not bit-identical to what an A30 would give. Gradient checkpointing is used for Qwen-7B and InternVL3-2B (as in A19) and not for Qwen-3B.
- **Part H (higher doses, A-OKVQA, length shortcut, Qwen2.5-VL-3B, seed 20261021):** d in {0.97, 0.99}, 2 runs, `scripts/a21_run.py`, arms of A21. Analysed together with the A21 runs at d = 0.25, 0.70, 0.92 of the same series.
- **Part P (position shortcut, A-OKVQA, Qwen2.5-VL-3B, seed 20261021):** the correct option is placed at letter A with probability d (exact realised rate, `scripts/dose_plant.py`), d in {0.25, 0.70, 0.92}, 3 runs, `scripts/a21_run.py --shortcut position`.
- **Part S (second seed of A19, PMC-VQA, length shortcut):** InternVL3-2B and Qwen2.5-VL-7B, seed 20261012, d in {0.25, 0.70, 0.92}, 6 runs, `scripts/dose_run.py`, arms and ladder as in A19.
Analysis script `scripts/a22_analysis.py`, hashed before the first run.

## Rules
- **H1 (further decline):** g(0.99) <= g(0.92) - 3.0 points, with g(0.92) taken from the A21 run of the same series.
- **H2 (cue uptake):** the shortest option is picked in at least 95% of E1 questions at d = 0.99.
- **H3 (lock-out reached, reported, not required):** g(0.99) <= 2.0 points and V*(0.99) = 7. **Expectation, not a rule:** H1 and H2 hold; H3 may fail if strong natural visual evidence keeps the image useful even when the shortcut is almost deterministic.
- **P1 (dose-response, position):** g(0.25) - g(0.92) >= 5.0 points and g(0.25) > g(0.70) > g(0.92).
- **P2 (cue uptake):** letter A is picked in at least 85% of E1 questions at d = 0.92 and in at most 35% at d = 0.25.
- **P3 (boundary shift):** V*(0.92) >= V*(0.25) + 1. **P4 (lock-out, reported):** g(0.92) <= 2.0 and V*(0.92) = 7.
- **S (second seed of A19):** the frozen rules A, B and C of A19 (`g_pred` = 13.5 / 3.4 / 0.1 points at d = 0.25 / 0.70 / 0.92, tolerance 4.0; V*(0.25) <= 6 and V*(0.92) = 7; monotone: g(0.25) - g(0.70) >= 2, g(0.70) - g(0.92) >= 1, g(0.92) <= 2), each model separately.
The accuracy-drop condition of A21's R2 is not used in A22 (the A21 runs show that on a benchmark with strong visual evidence the accuracy on unplanted questions does not measure cue use).

## Reading rule (fixed)
- H1 and H2 hold: the paper reports that the benefit keeps falling towards the almost-deterministic cue, and whether H3 holds locates the point where the image stops mattering for this model and benchmark; if H1 fails, the paper reports that the benefit plateaus and states the plateau as a boundary condition of the lock-out claim.
- P1-P3 hold: the paper reports that the second non-medical shortcut gives the same pattern; if any fails, it reports which.
- S: for each model, "reproduced in a second seed" only if A, B and C hold; one pass: stated for that model; none: the held-out claim is reduced to the seed that passed.
Results are reported whatever they are; no run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One general-domain benchmark; one seed in parts H and P; the hardware and CUDA version differ from the earlier runs; contamination of the pretrained models with COCO/A-OKVQA is possible; the part-H doses are chosen after seeing that d = 0.92 did not lock out (hence the rules above are fixed before the runs but the choice of doses was informed by A21).

## Amendments log
(A0-A22.)
