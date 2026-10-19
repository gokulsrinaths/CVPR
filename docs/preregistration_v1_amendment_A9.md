# Amendment A9 to pre-registration v1: controlled fine-tuning study (does training on this benchmark create the shortcut habit and the tie-breaker structure, in other models, on unseen cases?)

Appended 2026-10-03, before any training run exists. v1, A0-A8 unchanged. This is the "controlled fine-tuning study" proposed after the A6/A7 rewrite failed; it needs **no rewritten options**: the comparison is the same model before and after
fine-tuning on this benchmark's training split.

## What has been seen (not predictions)
For the MedReason adapter (one model, in-sample suspected on the pool): the shortcut statistics (A0-A5), the tie-breaker structure on FRESH3000 (A8, all six predictions supported) and on TEST711/DEV700 (exploratory). Zero-shot Qwen2.5-VL-7B with a letter-only generated answer
scored 28.0% on TEST711 and picked the shortest option 25.6% of the time (A4, not evaluable). **Not seen:** any output of the two new models after fine-tuning, and the zero-shot letter-logit readout below.

## Models and recipe (fixed; no tuning, no second run unless a crash)
- **Models:** `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen research licence, research use) and `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0), local, bf16, sdpa attention.
- **Training set TR:** official train MCQs minus the 886 test ids, minus the 62 same-source cases N(T), minus DEV700 (H2), POOL700 (J1) and FRESH3000 (A8 sample ids from its `config.json`). Its size is reported. The evaluation cases are never trained on.
- **LoRA:** r = 16, alpha = 32, dropout 0.05, on the language-model linear layers only (q, k, v, o, gate, up, down; the vision tower is frozen), adapter weights fp32.
- **Optimisation:** AdamW (betas 0.9/0.999, weight decay 0), learning rate 1e-4, linear warm-up over 3% of the steps then cosine to 0, **1 epoch**, per-device batch 1 with gradient accumulation 16, gradient clip 1.0, seed 20261003, shuffled with that seed.
- **Input/target:** system `You are a medical imaging expert.`; user = the image then the MCQ text of A4 (question, `Options:` with `  X: text` lines, a blank line, `Answer with only the letter (A, B, C, D or E) of the correct option.`); assistant = the correct letter. Loss only on the answer-letter token and the end-of-turn token.
  Images are downscaled (LANCZOS, never upscaled) so the longest side is at most 448 px, for training and evaluation. No modality detector, no MAP note.
- **Readout (all stages and models):** the log-probabilities of the five letter tokens at the position after the generation prompt, softmaxed over A-E (`direct`).
- If a job cannot finish within its time limit it is reported as not completed; no recipe change without an amendment. A short smoke run (40 optimiser steps, 20 evaluation cases) checks the harness only and produces no scientific result.

## Evaluation (per model)
- **Before training (zero-shot, same readout):** TEST711 arms `base` and `black`; VAL700 `base`.
- **After training:** VAL700 `base` (unlabeled, sets the model's own tau_m = the 15th percentile of its margins, index int(0.15 x 700), as in A1); TEST711 `base`, `black`; FRESH3000 `base`, `black`, `mismatch_rand`, `mismatch_nn` (same donors as A8).
- Strata, "changed", exact McNemar and intervals exactly as in A8, with the model's own tau_m and its own black-arm margins.

## Predictions (each falsifiable; each model separately; a statement is supported only if it holds for both models)
- **G1 (training creates the shortcut habit):** on TEST711 with the true image, the share of predictions equal to the shortest option (A0 character rule) is **at least 0.80 after fine-tuning and at most 0.50 before**. Falsified otherwise.
- **G2 (evaluable):** accuracy on FRESH3000 with the true image after fine-tuning is at least 0.80. Otherwise G3 and G4 are NOT EVALUABLE for that model (reported anyway).
- **G3 (tie-breaker structure replicates):** on FRESH3000, (C1) in HI the share of cases whose prediction changes between `base` and `black` is at most 0.01; (C2) in LO accuracy(base) - accuracy(black) is at least 5.0 points with exact McNemar p < 0.05 (not evaluable if LO has fewer than 50 cases); (C3) at least 90% of all
  base-vs-black changes occur outside HI; (C4) in HI the share changing between `base` and `mismatch_nn` is at most 0.02.
- **G4 (error alarm):** on FRESH3000 the cases with true-image margin below the model's tau_m make up 10% to 25% of the sample and contain at least 80% of the errors.
- **Reported without prediction:** accuracy before and after (TEST711, black arm), the headline true-vs-black gap, image influence on the log-odds, shortest-wrong subset, results by stratum, training loss curve and time.

## Reading rule (fixed)
G1, G2, G3 supported for both models => training on this benchmark produces the shortcut habit and the tie-breaker structure in two independent models on cases they never saw (the first labelled out-of-sample test of the structure). G3 failing for one model => the structure is not general and the paper says so. G1 failing => the
habit is not explained by this fine-tuning. Whatever happens is reported. Limits stated in advance: two models of one family, one recipe, one epoch, one benchmark.

## Amendments log
(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9.)
