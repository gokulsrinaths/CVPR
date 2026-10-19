# Amendment A21 to pre-registration v1: non-medical replication of the shortcut dose-response (A-OKVQA)

Appended 2026-10-04, before any run below. v1 and A0-A20 unchanged. Motivation: all dose-response evidence (A17, A19) is on one medical benchmark (PMC-VQA). A21 asks whether the same intervention produces the same pattern on a general-domain multiple-choice VQA benchmark whose images are natural photographs.

## What has been seen (not predictions)
Only dataset statistics. A-OKVQA (HuggingFaceM4/A-OKVQA, official train 17,056 and validation 1,145 questions; test labels are hidden), four options per question. The correct option is the unique shortest option in 15.8% of the training questions and 16.1% of the validation questions (chance with ties excluded: below 25%), the gold letter is uniform (24-26% per letter). No A-OKVQA question was passed through any model before this amendment: no zero-shot accuracy, no black-image accuracy and no fine-tuning result has been seen.

## Data (`scripts/a21_build.py`, hashes in `data/aokvqa/split_a21/SHA256.json`)
E1 = the 1,145 official validation questions (test set). From the training questions no question whose image also occurs in E1 exists (0 removed); the rest are grouped by image hash, ordered by sha256('A21|' + image hash), and whole groups fill VAL700 (700 questions, used only for the 15th-percentile text-saturation threshold, as in A11) and TR (9,277 training questions, the size of A11). The splits are image-disjoint (checked in the script). Question text and image are unchanged; only the options are changed by the planted shortcut.

## Runs
Recipe of A11 / A17 (LoRA r = 16, one epoch, learning rate 1e-4, maximum image side 448 px) with one change: the system prompt is "You are a helpful assistant that answers multiple-choice questions about images." instead of the medical one. Shortcut: option length by neutral filler on all options (`scripts/dose_plant.py`, unchanged, exact realised rate), planted in training and in evaluation options; **d in {0.25, 0.70, 0.92}**.
Series (each series has the three doses): **S1** Qwen2.5-VL-3B seed 20261021; **S2** Qwen2.5-VL-3B seed 20261022; **S3** InternVL3-2B seed 20261021 (gradient checkpointing). 9 runs, A30 GPUs. Arms: true image (base), black, and the blur ladder (radius 32, 16, 8, 4, 2) on E1; base on VAL700; original (unplanted) options with true and black image on E1. The mismatched-image arms and the downsampling ladder of A17 are not used (the nearest-neighbour donors were chosen with a medical image encoder). Runner `scripts/a21_run.py`, per-run summary `scripts/a21_summarize.py` (copies of `dose_run.py`/`dose_summarize.py` with the paths, prompt and arms above changed), analysis `scripts/a21_analysis.py`. Image benefit g, veto boundary V* (7 = none), text margin and the shortest-option pick rate are defined as in A17.

## Rules (each series is evaluated separately)
- **R1 (dose-response):** g(0.25) - g(0.92) >= 5.0 points and g(0.25) > g(0.70) > g(0.92).
- **R2 (cue uptake):** the shortest option is picked in at least 85% of E1 questions at d = 0.92 and in at most 35% at d = 0.25, and the accuracy on the original unplanted E1 questions with the true image is at least 5.0 points lower at d = 0.92 than at d = 0.25.
- **R3 (boundary shift):** V*(0.92) >= V*(0.25) + 1 (blur ladder).
- **R4 (lock-out, strong form):** g(0.92) <= 2.0 points and V*(0.92) = 7.
No numerical prediction of g is made (the image carries far more information in A-OKVQA than in PMC-VQA, so the values are not expected to match the medical curve). **Expectation, not a rule:** R1-R3 hold; R4 may fail, because strong visual evidence may partly override a 92% shortcut, which would be the visual-strength side of the claim and would be reported as such.

## Reading rule (fixed)
- **Replicates:** R1, R2 and R3 hold in all three series: the paper states that the dose-response of the shortcut was reproduced on a general-domain benchmark with a second model family (one benchmark, one shortcut, one seed for the second family), and reports R4 as it comes out.
- **Partly replicates:** R1-R3 hold for both Qwen series but not for InternVL3-2B (or the reverse): the paper states for which family it holds.
- **Does not replicate:** R1 fails in S1: the claim of the paper stays restricted to medical VQA, the A-OKVQA result is reported as a boundary condition (strong natural visual evidence may resist the shortcut; this explanation is a hypothesis, not tested) and the title and abstract keep the medical scope. If R1 holds but R4 fails, both are reported.
Results are reported whatever they are; no run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One general-domain benchmark, one shortcut (length), two to three series, one seed for InternVL3-2B; the pre-trained models may have seen COCO images and A-OKVQA-like questions (possible contamination of the visual side, not of the planted shortcut); the planted filler is the same neutral vocabulary as in the medical experiments; gradient checkpointing on InternVL3-2B; no downsampling ladder and no mismatched-image arms.

## Amendments log
(A0-A21.)
