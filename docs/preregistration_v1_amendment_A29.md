# Amendment A29 to pre-registration v1: does the image benefit without a cue predict the lock-out strength? A prediction for OmniMedVQA

Appended 2026-10-05, before any run below. v1 and A0-A28 unchanged. Motivation: the paper reports that the shortcut strength at which the image stops mattering differs by benchmark (PMC-VQA, A-OKVQA) and that the image benefit without a cue differs too (about +13.5 against +34 points), but two benchmarks cannot distinguish a rule from a coincidence. A third benchmark with a smaller benefit without a cue gives a written-down prediction.

## What has been seen (not predictions)
- Planted-shortcut results of Qwen2.5-VL-3B (A17, A21, A22): image benefit g at d = 0.25 / 0.70 / 0.92 is about +13.5 / +3.4 / +0.1 points on PMC-VQA (lock-out, g <= 2.0, between d = 0.70 and 0.85) and +33.9 (d = 0.25) and +8.1 (d = 0.92) on A-OKVQA, with lock-out only at d = 0.99.
- OmniMedVQA (A16; split `data/omnimed/split_a16`, E1 = 2,000 test questions, four options): without a planted cue the image benefit of the fine-tuned models is +5.7 (Qwen-3B), +7.1 (Qwen-7B) and +16.2 (InternVL3-2B) points; the correct option is the unique shortest in 22.2% of E1 questions and the models pick the shortest in 21.8%. No planted-shortcut run on OmniMedVQA exists.
- A28 (text-only audit): OmniMedVQA has no severe length cue.

## Runs
Recipe, planting code (`scripts/dose_plant.py`, exact realised rate), training set (TR, 9,277 questions), threshold set (VAL700), evaluation set (E1) and arms exactly as in A21, with the A16 medical system prompt; script `scripts/a29_run.py`, summaries `scripts/a29_summarize.py`, analysis `scripts/a29_analysis.py`, job file `slurm/41_a29_a30.sbatch` (all hashed before the first submission). Shortcut: length. Model: Qwen2.5-VL-3B. Doses: 0.25 (control carrying the filler without a cue), 0.50, 0.70, 0.85. Two series, seeds 20261091 (S1) and 20261092 (S2): 8 runs on A30 GPUs. Arms: true image, black, blur ladder (32, 16, 8, 4, 2 px) on E1; original unplanted options with true and black image on E1.

## Rules (each series separately; g = acc(true image) - acc(black), points, on E1)
- **L0 (the experiment is informative):** g(0.25) >= 3.0.
- **L1 (dose-response):** g(0.25) - g(0.85) >= 3.0 and g does not rise by more than 0.5 points from one dose to the next.
- **L2 (cue uptake):** the planted-option model picks the shortest option in at least 85% of E1 questions at d = 0.85.
- **L3 (lock-out prediction), tested only if L0-L2 hold in both series:** *main:* g(0.85) <= 2.0 in both series; *strong:* g(0.70) <= 2.0 in both series.
The visual veto boundary V* (A17 definition) is reported, not tested.

## Prediction (written before any run)
Because the benefit without a cue is smaller on OmniMedVQA (+5.7) than on PMC-VQA (+13.5), the lock-out should come at a weaker cue: L3-main holds, and the strong form is expected to hold (g(0.70) <= 2.0, against +3.4 on PMC-VQA). The ordering of lock-out strengths would then follow the ordering of the benefit without a cue: OmniMedVQA <= PMC-VQA (about 0.85) < A-OKVQA (0.99).

## Reading rule (fixed)
- L0, L1 or L2 fails in either series: not informative; the paper reports it and makes no lock-out claim for OmniMedVQA.
- L3-strong holds: the paper states that on three benchmarks the lock-out strength is ordered like the benefit without a cue (three points, one model, not a law).
- L3-main holds, L3-strong fails: the paper states the lock-out of OmniMedVQA is no earlier than that of PMC-VQA; the ordering OmniMedVQA < PMC-VQA is not supported.
- L3-main fails: the paper states that the benefit without a cue does not predict the lock-out strength (a failed prediction).
Results are reported whatever they are; no run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One model, one shortcut, doses 0.25 to 0.85 only (no 0.92 or 0.99), two seeds, a benchmark whose questions are partly answerable from text alone (black-image accuracy above 90% after fine-tuning), three benchmarks in total; correlation across three benchmarks cannot show that the benefit without a cue is the cause.

## Amendments log
(A0-A29.)
