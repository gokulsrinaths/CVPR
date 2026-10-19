# Amendment A16 to pre-registration v1: 7B seeds and a third benchmark (OmniMedVQA)

Appended 2026-10-04, before any run below. v1 and A0-A15 unchanged.

## Motivation
A14 replicated the primary condition over seeds for Qwen2.5-VL-3B only. A16 adds (S) two more seeds of the 7B model on both benchmarks and (O) a third benchmark, OmniMedVQA (Hu et al., CVPR 2024), to test whether the text-locked / contestable pattern found on Med-CMR, PMC-VQA and the planted-cue variant generalises, using a simple rule fitted on those conditions and evaluated out of sample.

## What has been seen (not predictions)
All A8-A15 results. For OmniMedVQA only counts from the data files: the open-access subset has 88,996 QA items on 82,059 images from 42 source datasets (RadImageNet 56,697 items), 82,082 items with four distinct options and a gold option among them, and the correct option is the unique shortest in 27.0% of these (chance 25%). After the split was built (before any model output): the correct option is the unique shortest in 22.2% of E1, 21.3% of E2 and 22.3% of TR (below the 25% chance rate, so the sampled benchmark has no length cue); E1 covers all 42 source datasets and has 872 disease-diagnosis, 458 anatomy, 378 modality-recognition, 184 other-attribute and 108 lesion-grading questions. No model output on OmniMedVQA has been seen. Seed-1/2 outputs of the 7B models have not been seen.

## Execution note
The H100 queue is saturated (estimated start of a new job: 2026-10-06). All jobs of this amendment run on NVIDIA A30 (24 GB) GPUs of the same cluster (partition beartooth-gpu), bf16, batch size one. The 7B models and InternVL3-2B do not fit in 24 GB with the recipe unmodified (smoke tests: InternVL3-2B ran out of memory), so **all training jobs of this amendment use gradient checkpointing** (`--gc`, non-reentrant), which computes the same gradients with a different memory schedule; other recipe elements are unchanged. Seed 0 of the 7B models (A9, A11) was trained on an H100 without checkpointing; bf16 results can differ in the last digits between GPU types.

## Part S: 7B seed replication
Qwen2.5-VL-7B, A9 recipe on Med-CMR (`scripts/ft_run6.py`) and A11 recipe on PMC-VQA (`scripts/pmc_run4.py`), seeds 20261004 and 20261005 (seed 0 = A9/A11). Analyses and predictions as in A14: **S1 (Med-CMR)** R1-R4 of A10 (analysis `ft_summarize2.py`); **S2 (PMC-VQA)** Q1-Q3 of A11 (analysis `pmc_summarize.py`). Reported: mean and standard deviation over three seeds.

## Part O: OmniMedVQA
**Data (`scripts/omni_build.py`, seed 20261004, ids/records/donors hashed in `data/omnimed/split_a16/SHA256.json`).** Open-access subset, four-option questions with a unique gold. Sampling frame: per source dataset, groups (an image; for datasets with patient folders, the patient) are shuffled and taken until 1,200 questions per dataset are reached (so no dataset dominates). From the frame, groups are assigned in a shuffled order to **E1** (2,000 questions, primary), **E2** (3,000), **VAL700** (700, unlabeled, sets tau_m) and **TR** (9,277, fine-tuning); all four are disjoint by group. Claims are image-disjoint (patient-disjoint only where a patient folder exists), not patient-disjoint in general; there is no official test split. Counterfactual images for E1 and E2: black 224x224; random (derangement within the set, other group); nearest neighbour (PubMedCLIP cosine, same set, other group).
**Systems and recipe.** Qwen2.5-VL-3B, Qwen2.5-VL-7B (gradient checkpointing), InternVL3-2B; the A11 recipe (LoRA r = 16, one epoch over TR, lr 1e-4, batch 16, seed 20261003, images at most 448 px, letters A-D) (`scripts/omni_run.py`). Evaluation: zero-shot (VAL700 and E1, arms base, black, random, nearest neighbour); fine-tuned: VAL700 base; **E1** with arms base, black, random, nearest neighbour, blur8, blur24 (Gaussian blur radius 8 and 24 px on the 448 px image); E2 with base, black, random, nearest neighbour. Analysis `scripts/omni_summarize.py` (A11 analysis plus the quantities below).
**Predictions (each system, on E1; g(I) = accuracy(I) - accuracy(black); M_T = black-image log-odds margin of A15):**
- **O1 (the habit does not exceed the cue):** the share of E1 predictions equal to the unique shortest option is at most (the share of E1 questions whose correct option is the unique shortest) + 0.10.
- **O2 (lock rule, fitted on A9/A11/A13/A15 conditions, tested out of sample):** the all-question gap g(true) is at least +5.0 points **if and only if** the median of M_T over E1 is below 1.5 log-odds (conditions seen: Med-CMR median M_T 8.5-10.6 with gap about 0; PMC-VQA 0.25-0.38 with gap +14 to +15; planted cue 3.25-3.4 with gap about 0).
- **O3 (pattern under blur):** if g(true) >= +5.0 then g(blur24) < g(blur8) < g(true); otherwise |g(I)| <= 1.0 point for I in {blur24, blur8, true}.
- **O4 (no veto):** if at least 100 E1 questions have a black-image probability margin >= 0.99, at most 2% of their answers change between the black and the true image; otherwise not evaluable.
- **Reported without prediction:** zero-shot and fine-tuned accuracy, strata (tau_m rule of A8), accuracy by question type (disease diagnosis, anatomy, modality recognition, other attributes, lesion grading), error alarm, E2 results, training loss.

## Reading rule (fixed)
O2 and O3 hold for all three systems: the lock rule and the blur pattern generalise to a third benchmark. Otherwise the paper reports which fail and the observed medians and gaps, and does not revise the rule. A failed prediction is reported with the data that falsified it.

## Limits (stated in advance)
OmniMedVQA questions are templated and many options are dataset-specific class names, so text priors may be strong in ways unrelated to option length; the sampling frame is capped per dataset; open-access subset only; one seed per system; the images come from public datasets with their own licenses (research use, no redistribution); A30 versus H100 numerics; gradient checkpointing for the 7B runs.

## Amendments log
(A0-A16.)
