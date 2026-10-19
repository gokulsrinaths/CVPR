# Amendment A20 to pre-registration v1: exploratory follow-up, a tempered bias expert for shortcut-rescue on Med-CMR

Appended 2026-10-04, before any run below. v1 and A0-A19 unchanged. **Exploratory:** one pre-specified change, run once, reported whatever happens; not part of any confirmatory family.

## Motivation and what has been seen
A18: shortcut-rescue (product-of-experts) training restores visual control in the controlled setting (PMC-VQA, planted length cue) but **collapsed to chance accuracy on Med-CMR** (M-F not supported; accuracy 19.8-21.5%). The text-only bias expert of A12 on Med-CMR is extremely confident (median probability margin 0.9995, i.e. log-odds of about 8-10), whereas the experts of the planted PMC-VQA setting where the method worked have median log-odds margins of about 3. With a saturated expert the main model receives almost no gradient from the questions the expert solves. Not seen: any run with a tempered expert.

## Change (single, fixed, no tuning)
Bias log-probabilities are divided by a temperature **T = 3** and renormalised (the expert's log-odds margin falls to about 3, comparable to the settings where the method worked); everything else is as in A18 (Qwen2.5-VL-3B, A9 recipe and data, A12 cross-fitted expert log-probabilities, w = 1, product-of-experts loss, main model used alone at test time). Seeds 20261011-13 (`scripts/ft_run8.py`, derived from `scripts/ft_run7.py`), analysed as M-F of A18 (`scripts/a12_summarize.py m2`).

## Prediction (exploratory)
**M-F':** in at least two of three seeds, on the A9-LO set (the 410 FRESH questions that are text-undecided for the A9 3B model) the image effect is >= +5.0 points with p < 0.05 and overall FRESH3000 accuracy is >= 93.5% (within 2.0 points of 95.5%).

## Reading rule
M-F' holds: reported as an exploratory single-setting result, to be confirmed. M-F' fails: reported as a further negative result; the limit stated in A18 stands (the remedy is validated in the controlled setting, not on Med-CMR). T is not changed afterwards.

## Limits
Exploratory; one temperature; one model; one real shortcut; A30 numerics.

## Amendments log
(A0-A20.)
