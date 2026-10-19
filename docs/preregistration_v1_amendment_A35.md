# Amendment A35 to pre-registration v1: where in the image does the evidence sit?

Appended 2026-10-05, before any run below. v1 and A0-A34 unchanged. Motivation: the only visual manipulation in the paper is global blur; a vision reviewer will ask whether the image benefit that remains without a cue rests on the central content of the image or on its periphery. A35 is an evaluation-only test on saved adapters.

## What has been seen (not predictions)
Image benefit at d = 0.25 (control) of Qwen2.5-VL-3B: about +13.5 points on PMC-VQA (A17; three seeds 20261011-13) and +33.9 on A-OKVQA (A21; seeds 20261021-22); at d = 0.70 on PMC-VQA +3.4. Blur results exist; no occlusion result exists.

## Runs (`scripts/a35_occl_eval.py`, `slurm/45_a35_occl.sbatch`, `scripts/a35_analysis.py`; all hashed before the first submission)
Evaluation only, 8 saved adapters: PMC-VQA d = 0.25 and d = 0.70 (seeds 20261011, 20261012, 20261013) and A-OKVQA d = 0.25 (seeds 20261021, 20261022); E1 (1,999 or 1,145 questions) with the planted options of the adapter's training, and the arms true image, black image, **centre occluded** (the central rectangle holding half of the image area set to black) and **border occluded** (the complementary frame, half of the area, set to black).

## Rules
- **O1 (PMC-VQA, d = 0.25):** accuracy with the centre occluded is lower than with the border occluded by at least 2.0 points in at least two of the three seeds.
- **O2 (A-OKVQA, d = 0.25):** the same difference of at least 2.0 points in both seeds.
Reported, not tested: the share of the true-minus-black benefit lost under each occlusion, and the d = 0.70 adapters.

## Reading rule (fixed)
- O1 and O2 hold: the paper states that the image evidence these models use sits mainly in the central half of the image in both benchmarks (a coarse localisation, not an attention analysis).
- O1 holds, O2 fails, or the reverse: the paper states for which benchmark it holds.
- Neither holds: the paper states that no centre-versus-border preference was found at this resolution.
No evaluation is repeated or tuned after seeing its result.

## Limits (stated in advance)
Medical and natural images are usually centred, so a central preference may reflect composition, not diagnostic content; two region definitions only; one model size; evaluation only, no training; the number of occlusion arms is small.

## Amendments log
(A0-A35.)
