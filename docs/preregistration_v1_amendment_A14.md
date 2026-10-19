# Amendment A14 to pre-registration v1: seed replication and threshold sensitivity

Appended 2026-10-03, before any run below. v1 and A0-A13 unchanged. Motivation: an outside simulated review of the draft asked for (i) replication of the primary controlled condition across random seeds and (ii) a sensitivity analysis of the text-saturation thresholds. Both are specified here before any seed-1/2 output or sensitivity table has been seen.

## What has been seen (not predictions)
All results of A8-A12 and the (not yet finished) A13 and weighted-training runs are not used here. Not seen: any output of the seed runs below, or any sensitivity table.

## Part S: seed replication (Qwen2.5-VL-3B, the primary controlled condition)
The A9 (Med-CMR) and A11 (PMC-VQA, original options) recipes are rerun with two further seeds (20261004 and 20261005; the earlier runs, seed 20261003, are "seed 0"). The seed sets the training order, the LoRA initialisation and the dropout (`torch.manual_seed`); the data, recipe and evaluation sets are unchanged. Systems and jobs: Med-CMR 3B x 2 seeds (`scripts/ft_run5.py`, evaluated on VAL700, TEST711, FRESH3000 with the four image arms; analysed with the A10 analysis `ft_summarize2.py`), PMC-VQA 3B x 2 seeds (`scripts/pmc_run3.py`; analysed with the A11 analysis `pmc_summarize.py`). Predictions (each seed, each benchmark):
- **S1 (Med-CMR, A10's R1-R4):** R1 the image effect on the text-undecided stratum is not (>= +5.0 points with p < 0.05); R2 the share of TEST711 predictions equal to the shortest option >= 0.80; R3 the share of text-decisive (HI) questions whose prediction changes under a black image <= 0.01; R4 the flagged questions are 10-25% of FRESH3000 and contain >= 80% of the errors.
- **S2 (PMC-VQA, A11's Q1-Q3):** Q1 the share of E1 predictions equal to the unique shortest option <= 0.35; Q2 the all-question true-vs-black gap on E1 >= +5.0 points with p < 0.05; Q3 the effect in the least-confident stratum >= +5.0 points with p < 0.05.
- **Reported:** mean and standard deviation over the three seeds (seed 0 included) of the all-question gap, the undecided-stratum effect, the shortest-option share, accuracy.

## Part T: threshold sensitivity (CPU only; `scripts/a14_threshold.py`)
For the seven systems of A8, A9 and A11 (released adapter, Qwen 3B and 7B, InternVL3-2B on Med-CMR; Qwen 3B and 7B, InternVL3-2B on PMC-VQA) on FRESH3000 (Med-CMR) or E1 (PMC-VQA): text-decisive cut-offs on the black-image margin of 0.5, 0.9, 0.99 and 0.999 (the number of questions above and the number whose prediction changes under the black, the random and the nearest-neighbour image), and text-undecided cut-offs of 0.5, 0.7 and 0.9 (number of questions and true-minus-black effect with exact McNemar p). Predictions on Med-CMR:
- **T1:** for every system and every text-decisive cut-off >= 0.9 that holds at least 100 questions, the share of answers changed by each of the three counterfactual images is <= 0.02.
- **T2:** for the three fine-tuned systems and every undecided cut-off that holds at least 100 questions, the effect is not (>= +5.0 points with p < 0.05).
- **T3:** for the released adapter, the effect at the 0.9 cut-off is >= +5.0 points with p < 0.05 (A8: +12.1).

## Reading rule (fixed)
All S and T predictions hold: the Med-CMR/PMC-VQA contrast and the no-veto finding are not artefacts of one seed or one threshold, and the paper reports the spread. Any seed with a falsified prediction is reported as such, with the spread; a falsified T1 or T2 narrows the claim to the cut-offs where it holds.

## Limits (stated in advance)
Three seeds is a small sample; the seeds change initialisation, order and dropout but not the data split; sensitivity uses absolute cut-offs on the model's own black-image margin, not an exhaustive search.

## Amendments log
(A0-A14.)
