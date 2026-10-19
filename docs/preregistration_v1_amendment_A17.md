# Amendment A17 to pre-registration v1: shortcut dose-response and the visual veto boundary

Appended 2026-10-04, before any run below. v1 and A0-A16 unchanged.

## Question
How does the strength of a nonvisual shortcut interact with the strength of visual evidence to determine whether vision can affect a model's decision? Claim under test (not assumed): predictive nonvisual shortcuts progressively suppress visual control; visual evidence helps while the decision remains contestable, and can cease to affect decisions once shortcut-driven text evidence locks them. No claim of necessity or of a universal law is made or tested.

## What has been seen (not predictions)
A11 (PMC-VQA, original options, cue rate 19.7-25%): image gap +13.7 (Qwen 3B). A13 (cue planted at 92%): gap +0.05. A15 (two blur radii): on PMC-VQA original g(blur24) = +0.6, g(blur8) = +2.2, g(intact) = +13.2 (Qwen 3B), and about 0 at every level for the planted-cue and Med-CMR models. A15 competition-score results. **Not seen:** any model trained with a cue strength between 25% and 92%, any model trained with the position shortcut, any output on a six-level blur ladder or on downsampling, any run with the new seeds. The A13 92% runs used seed 20261003 and are not part of the confirmatory set.

## Shortcut 1: option length (neutral filler; `scripts/dose_plant.py`, filler of A12/A13; dose-controlled)
All four options of every question carry neutral filler words (A12/A13 mechanism); within each question set (TR, VAL700, E1) questions are ranked by a hash and a question is "cued" iff its rank-quantile u < **d** (the dose), so the realised cue rate equals d to within one question and the cued set is nested across doses; a cued question gets the correct option as the shortest, an uncued one a uniformly random other length rank. Doses **d in {0.25, 0.40, 0.55, 0.70, 0.85, 0.92}**. d = 0.25 is the control: every option carries filler, ranks are uniform, the cue is absent. Training and evaluation options are planted at the same dose (matched); each model is additionally evaluated on the original, unplanted E1 options. Achieved cue rates are validated (selftest and `config.json`; must equal d to within 0.002 on TR and E1 or the run is not started).
## Shortcut 2: answer position (`scripts/dose_plant.py`)
Option texts are unchanged; with the same rank-quantile assignment the correct option is placed at position A iff u < **p** and otherwise at a uniformly random position among B-D. Doses **p in {0.25, 0.55, 0.92}** (p = 0.25 is the formatting/tokenisation control: identical reordering code, uniform positions, no cue). No filler, no length change, no semantic content.

## Common design (frozen)
PMC-VQA split of A11 (`data/pmcvqa/split_a11`): TR 9,277 (training), VAL700 (unlabeled; sets tau_m by the A8 rule), E1 1,999 (primary evaluation). Qwen2.5-VL-3B, the A11 recipe (LoRA r = 16, one epoch, lr 1e-4, batch 16, images at most 448 px, letters A-D), `scripts/dose_run.py`, run on A30 GPUs (A15/A16 execution note: H100 queue saturated); the 3B model fits in 24 GB and is trained **without** gradient checkpointing (smoke-tested); the held-out models of A18 will use gradient checkpointing if they do not fit. **Seeds 20261011, 20261012, 20261013** for the length shortcut at all six doses (18 runs); the position shortcut at three doses with seed 20261011 (3 runs). Evaluation arms on planted E1: true (base), black, random image, nearest-neighbour image, and a **blur ladder** (Gaussian blur radius 32, 16, 8, 4, 2 px on the 448 px image) giving six visual levels V1..V6 (V6 = intact) plus the black reference; for the runs d in {0.25, 0.70, 0.92} of seed 20261011 also a **downsampling ladder** (longest side 8, 16, 32, 64, 128 px, bicubic, resized back to the original size). Degradations act on the image only (prompts are identical across arms; asserted in the selftest).

## Endpoints (per run, on planted E1)
accuracy by arm; **image benefit g(L) = accuracy(L) - accuracy(black)** (points) with exact McNemar p and a paired bootstrap 95% CI (2,000 resamples, seed 20261011); black-image margin distribution (median log-odds margin M_T, share with probability margin >= 0.99); share of predictions changed relative to the black-image answer (override rate) per arm; shortest-option (or position-A) pick rate; image benefit in the text-undecided stratum (A8 rule); accuracy and gap on the original E1 options.
**Visual veto boundary (criterion fixed now).** V* = the lowest ladder level L (V1 < V2 < ... < V6) with g(L) >= **2.0 points** and bootstrap 95% lower bound >= **1.0 point**; V* = 7 ("beyond the available evidence") if no level qualifies.

## Hypotheses (confirmatory; length shortcut, Qwen 3B, 18 runs)
- **H1 (dose-response of image benefit):** pooled Spearman rho(d, g(intact)) <= -0.70 with a one-sided permutation p-value (10,000 permutations of d within seeds) that survives Holm correction across H1-H4 (alpha = 0.05); within every seed rho <= -0.70; in every seed g(d = 0.25) >= +8.0 and g(d = 0.92) <= +2.0.
- **H2 (boundary shift):** rho(d, V*) >= +0.70 (same permutation test, Holm); in every seed V*(0.25) <= 6 and V*(0.92) = 7.
- **H3 (text margin tracks dose, and is the simpler predictor):** (a) rho(d, median M_T) >= +0.70 (Holm); (b) leave-one-seed-out linear prediction of g(intact): RMSE using the median M_T <= 1.25 x RMSE using the dose d. If (b) holds, text saturation is retained as the predictor of visual control and no further boundary/competition measure is proposed; the veto boundary is reported as a description, not a new predictor.
- **H4 (override map):** rho(d, override rate of the intact image) <= -0.70 (Holm); for every run with d <= 0.55 the override rate rises monotonically with ladder level (Spearman across V1..V6 >= +0.8).
- **H5 (second degradation family, d in {0.25, 0.70, 0.92}, seed 20261011):** V*_down(0.25) <= 6 and V*_down(0.92) = 7.
- **Second shortcut (position; seed 20261011):** P1 g(intact) at p = 0.92 <= +2.0; P2 g(intact) at p = 0.25 >= +8.0 (the control); P3 g(0.25) >= g(0.55) >= g(0.92) - 1.0; P4 V*(0.92) = 7.
All other quantities (accuracy, strata, shortest-pick rate, orig-E1 accuracy) are reported without prediction.

## Sequential stopping rule (to avoid expensive expansion)
Batch 1 = the six length doses at seed 20261011 (plus the downsampling arms of the three runs above). **If g(intact)(d = 0.25) - g(intact)(d = 0.92) < 5.0 points in batch 1, all further runs of this amendment are cancelled** and the negative result is reported. Otherwise batch 2 = seeds 20261012 and 20261013 (twelve runs) and the three position-shortcut runs.

## Held-out prediction (A18, to be written before any run)
After the Qwen-3B dose-response, a further amendment (A18) will fix, **before any InternVL3-2B or Qwen-7B dose-response run is started**, the quantitative expected g(intact) and V* for those models at d in {0.25, 0.70, 0.92}, from the Qwen-3B fit, together with the tolerance for success, and will be hashed. Hysteresis (low-high-low schedule) is not part of A17; it would be exploratory and is only considered if H1 and H2 hold, under its own pre-registration.

## Exclusions and procedure
A run that crashes or reports a non-finite loss is rerun with the same seed; no run is excluded or replaced because of its result. Realised cue rates that differ from d by more than 0.002 abort the run. Tests: exact McNemar, paired bootstrap, Spearman with stratified permutation tests, Holm over H1-H4; H5 and the position-shortcut predictions are reported with their own thresholds (single seed, not in the Holm family).

## Reading rule (fixed)
H1 and H2 both hold, and P1-P3 hold: the manuscript is restructured around "predictive nonvisual shortcuts can suppress visual control; visual evidence helps while the decision remains contestable and can cease to affect decisions once shortcut-driven text evidence locks them", as a controlled-intervention result for two shortcuts, one model family and one benchmark. If H1 holds and H2 or P-predictions fail, the dose-response is added as a section and the claim is limited to what held. If H1 fails, the manuscript keeps its structure and the negative result is reported in the supplement.

## Limits (stated in advance)
One benchmark and one model family for the confirmatory part; the shortcut is planted in training and evaluation options (sufficiency, not necessity of natural shortcuts); blur and downsampling are two of many ways to weaken visual evidence and also induce distribution shift; one seed for the second shortcut; A30 versus H100 numerics (all A17 runs on A30); the veto criterion thresholds (2.0 and 1.0 points) are conventions.

## Amendments log
(A0-A17.)
