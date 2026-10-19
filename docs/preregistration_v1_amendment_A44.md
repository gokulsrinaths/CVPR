# Amendment A44 to pre-registration v1: how fast does training on the benchmark's own data install the shortcut?

Appended 2026-10-18, before any run below. v1 and A0-A43 unchanged. Motivation: benchmark scores can rise through training on benchmark data without the ability (leakage and overfitting to a benchmark are documented for LLMs and vision-language models). Our paper studies the legitimate version, the benchmark's own training split: a zero-shot Qwen2.5-VL-7B is at chance on Med-CMR with or without the image (A40) while models fine-tuned on its training split reach the blind ceiling. A44 measures the learning curve of that effect with one model size and one recipe.

## What has been seen (not predictions)
A9 (Qwen2.5-VL-3B, one epoch on the 9,277 training MCQs, seed 20261003): accuracy on the 711 held-out MCQs 95.4% with the true image and about 95.6% with a black image; zero-shot A9 rows (`rows_zeroshot.jsonl`, same set) exist for the pre-training point. A40 (7B zero-shot): 27.6 / 27.7% true / black. The shortest-option pick rate of A9 after fine-tuning is 91-92% (before 23-25%). No fine-tuning on fewer than 9,277 questions has been run.

## Runs (`scripts/a44_ft_subset.py`, a copy of `ft_run.py` with `--n-train` and `--seed`; `slurm/51_a44_subset.sbatch`; `scripts/a44_analysis.py`; all hashed before the first submission)
Qwen2.5-VL-3B, the A9 recipe unchanged (LoRA r = 16, one epoch, learning rate 1e-4, images at most 448 px, loss on the answer letter), trained on the first N questions of the A9 training order (the seeded shuffle of the 9,277 training MCQs, seed 20261003 or 20261004): N = 100, 300, 1,000 and 3,000, two seeds each (8 runs, A30 GPUs). Evaluation on the 711 held-out MCQs with the true and the black image. N = 0 is the zero-shot A9 evaluation and N = 9,277 the A9 run (seed 20261003 only).

## Rules
- **L1 (fast installation):** the black-image accuracy on the held-out MCQs is at least 80.0% after training on N = 1,000 questions, in both seeds.
- **L2 (monotone):** along N = 0, 100, 300, 1,000, 3,000 (and 9,277 for the first seed) the black-image accuracy never falls by more than 2.0 points from one N to the next, in each seed.
- **L3 (no headroom once installed):** the true-image minus black-image accuracy is at most 2.0 points for every N >= 1,000 (both seeds and A9).
Reported, not tested: true-image accuracy, shortest-option pick rate and headroom at every N.

## Reading rule (fixed)
- L1, L2 and L3 hold: the paper states that training on the benchmark's own data installs the shortcut within about 1,000 questions and leaves no headroom, whereas zero-shot models have only the knowledge route; one rule fails: the paper states which and reports the curve.
No run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One model size and recipe; two seeds; one benchmark; the training order is a seeded shuffle, so the first N questions are a random sample, not a curriculum; this is the benchmark's own training split (legitimate use), not leakage of test items.

## Amendments log
(A0-A44.)
