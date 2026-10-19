# Amendment A11 to pre-registration v1: a second benchmark without an option-length cue, and a third model family

Appended 2026-10-03, before any model has been run on the data below. v1 and A0-A10 unchanged.

## Motivation
A9/A10 found, on Med-CMR (where the correct option is the shortest in 91.9% of questions), that fine-tuned models become text-saturated (no image changes the answer when the text is decisive), pick the shortest option 92% of the time, and gain nothing reliable from the image on text-undecided questions (six recipes, all within -4.1..+2.6 points). Three limits are open: one benchmark, one model family, and no evidence on whether the cue is what drives the pattern. A11 adds (a) PMC-VQA, a four-option multiple-choice benchmark of figures from PubMed Central, in which no length cue exists, and (b) a third model family (InternVL3-2B).

## What has been seen (not predictions)
- Counts only, from the data files, before this amendment: the correct option is the unique shortest option in **19.7%** of the 2,000 questions of `test_clean.csv` and in **25.05%** of the 176,934 usable `train.csv` questions (chance for four options is 25%). No model output on PMC-VQA has been seen. Train and test_clean share no source article (0 train rows in the 1,145 test articles).
- Smoke runs (20 questions per group, 640 training examples, to test the code; their numbers are not used) were run on all three systems before this freeze: all ran, letter probability mass was >= 0.99, and tau_m on those 20 validation questions was about 0.07-0.12, i.e. PMC-VQA margins are far from saturated. This is why the saturation band (m_b >= 0.99) and the fixed-threshold dose-response are specified below next to the tau_m strata.
- A9/A10 results on Med-CMR for Qwen2.5-VL (see those amendments). **Not seen:** any InternVL3 output on any data.

## Data (frozen by `scripts/pmc_build.py`, seed 20261003; ids, records and donor maps hashed in `data/pmcvqa/split_a11/SHA256.json`)
Source: PMC-VQA (`xmcmic/PMC-VQA`, CC BY-SA 4.0): `test_clean.csv`, `train.csv`, `images.zip`, `images_2.zip`. Usable question: four non-empty distinct options, answer label in A-D, image available. All splits are **disjoint by source article** (PMC id prefix of the figure name), and train articles that occur in `test_clean` are removed.
- **E1** = the 1,999 usable questions of `test_clean.csv` (the official clean test; one of its 2,000 rows has duplicated options and is dropped). **Primary** evaluation set.
- **E2** = 3,000 questions from `train.csv` articles never used otherwise. **Replication** set (some of its images are compound figures; E1 is cleaner).
- **VAL700** = 700 further questions, labels never used, setting each model's tau_m (15th percentile of its true-image margins, as in A8).
- **TR** = 9,277 further questions, the fine-tuning set (same size as A9).
Counterfactual images for E1 and E2: black 224x224; random (fixed derangement within the set, other article); nearest neighbour (PubMedCLIP cosine top-1 within the set, other article; **no modality restriction**, unlike Med-CMR).

## Systems and recipe (frozen; identical to A9 except where stated)
Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct and InternVL3-2B-hf (`OpenGVLab/InternVL3-2B-hf`, transformers-native). Each: zero-shot evaluation, then LoRA r=16, alpha=32, dropout 0.05 on the language-model linear layers, AdamW lr 1e-4, 3% warm-up, cosine, **1 epoch** over TR, batch 1 with 16 accumulation, clip 1.0, seed 20261003, images at most 448 px, system "You are a medical imaging expert.", loss on the answer letter and end token only; letters A-D. One run per system, no tuning, no repeat unless a crash. Readout, margin (top-1 minus top-2 probability over the option letters), strata and tests as in A8/A9 (exact McNemar, Wilson/Clopper-Pearson). Strata: HI if m_b >= tau_m, else MID if m_b >= 0.90, else LO (so if tau_m < 0.90 the MID stratum is empty and LO is m_b < tau_m); a fixed-threshold dose-response (m_b < 0.5, 0.5-0.9, 0.9-0.99, >= 0.99) is reported as well, and the **saturation band** is m_b >= 0.99 irrespective of tau_m.

## Part 1: PMC-VQA, three fine-tuned systems. Predictions (each system, on **E1**; E2 is replication and reported alongside)
- **Q1 (the habit is benchmark-specific):** the share of E1 predictions equal to the unique shortest option is at most **0.35** (it was 0.92 on Med-CMR). Falsified if above 0.35.
- **Q2 (the image matters when there is no cue):** accuracy(base) - accuracy(black) over all of E1 is at least **+5.0 points** with exact McNemar p < 0.05.
- **Q3 (the retest of A9's G3 without the cue):** in the LO stratum of E1, accuracy(base) - accuracy(black) is at least **+5.0 points** with p < 0.05.
- **Q4 (no veto in the saturation band):** among E1 questions with black-image margin >= 0.99, the share whose prediction changes between base and black is at most **0.02** (not evaluable if the band holds fewer than 100 questions).
- **Q5 (error alarm transfers):** the error rate among questions flagged (true-image margin < tau_m) is at least **1.5x** the error rate among unflagged questions, and the AUROC of the (negative) true-image margin for detecting errors is at least **0.65**.
- **Reported without prediction:** zero-shot and fine-tuned accuracy; all arms (black, random, nearest-neighbour) by stratum and by fixed threshold; strata sizes; training time and loss; E2 results.

## Part 2: InternVL3-2B on Med-CMR (third family, same data and protocol as A9/A10)
`scripts/ft_run3.py` trains on A9's TR (9,277 Med-CMR MCQs) and evaluates VAL700, TEST711 and FRESH3000 exactly as A9. Predictions are A10's **R1-R4** (R1: LO image effect below +5.0 points or p >= 0.05; R2: the share of TEST711 predictions equal to the shortest option >= 0.80; R3: HI answer-change share <= 0.01; R4: flagged 10-25% of FRESH3000 and capturing >= 80% of the errors), and
- **R5 (no veto under wrong images):** in HI the share of FRESH3000 answers changed by the random and by the nearest-neighbour image is at most 0.01 each.

## Reading rules (fixed)
- If Q1 holds and Q2/Q3 hold for the models of Part 1, while R1/R2 hold in Part 2: across the two benchmarks, the length cue goes with both the shortcut habit and the weak image use. This is a **between-benchmark contrast, not a causal test** (the benchmarks differ in more than the cue: topic, image type, question style, size), and the paper says so.
- If Q1 holds but Q3 fails: the habit is benchmark-specific but the weak image benefit on text-undecided questions is not explained by the cue; the paper reports it as a property of the fine-tuned models.
- If Q4 fails: the "no veto" finding is specific to Med-CMR's saturation, and the paper says so and replaces the general claim by the dose-response.
- Any other combination is reported as it is. No prediction is revised after results are seen.

## Limits (stated in advance)
Four-option questions (chance 25%, not 20%); PMC-VQA questions are partly generated; compound figures in E2; one seed per system; the NN arm is not modality-restricted on PMC-VQA; PubMedCLIP was trained on PubMed-like data and may overlap with PMC figures; MedGemma could not be used (gated).

## Amendments log
(A0-A11.)
