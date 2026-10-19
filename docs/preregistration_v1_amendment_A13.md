# Amendment A13 to pre-registration v1: planting a length cue in a benchmark that has none

Appended 2026-10-03, before any run below. v1 and A0-A12 unchanged.

## Motivation
Three attempts to test the length cue by removing it from Med-CMR failed to give a causal answer: two LLM rewrites failed their gates (A6, A7); filler padding (A12) did not remove the cue for the learner (models trained on options with randomised surface length still picked the originally shortest option 91% of the time, and accuracy and image indifference were unchanged), and it harmed evaluation-time accuracy by 6-10 points (L1 not evaluable). A13 tests the cue in the other direction: plant it in PMC-VQA, where models use the image (A11), and ask whether the behaviour of Med-CMR appears.

## What has been seen (not predictions)
A11 (PMC-VQA, fine-tuned models, original options) and A12 results. The share of PMC-VQA questions whose correct option is the unique shortest is 19.7% (E1) and 25.05% (train). Not seen: any output of a model trained or evaluated on planted options.

## Intervention (`scripts/len_implant.py`, hashed)
All four options of every question are extended by neutral filler words (the filler of `len_rand.py`, A12) so that the final option lengths are exactly s_max + 4j characters (j = 1..4); every option carries filler. The correct option receives the shortest length (j = 1) with probability **0.92**, otherwise a uniformly random other rank; wrong options take the remaining ranks in random order (seeded by case id). The planted cue is therefore the same strength as in Med-CMR (91.9%) and is carried by the surface length, the quantity the model can read directly.

## Design (frozen)
Systems: Qwen2.5-VL-3B, Qwen2.5-VL-7B, InternVL3-2B. Each is fine-tuned with the A9/A11 recipe (same data, order, seed as A11) on the **planted** training set TR (9,277 questions), and evaluated, with true and black images, on E1 planted, E1 original and E2 planted; VAL700 planted (unlabeled) sets tau_m. The comparison models are the A11 fine-tuned models of the same family (original options).

## Predictions (each system, on E1)
- **I1 (the habit is learned):** the share of E1-planted predictions equal to the unique shortest option is >= 0.80.
- **I2 (text becomes decisive):** the share of E1-planted questions with a black-image margin >= 0.99 is >= 0.30 (A11: 0).
- **I3 (the image gap shrinks):** accuracy(true) - accuracy(black) over all E1-planted questions is <= half of the A11 gap of the same family (A11: +13.7, +14.9, +14.4 points).
- **I4 (the cue is used):** accuracy on **E1 original** (cue absent) is at least 5.0 points lower than the A11 accuracy of the same family (A11: 51.8%, 54.1%, 51.0%).
- Reported without prediction: accuracy on planted questions, strata, undecided-stratum effect, E2 results, training loss.

## Reading rule (fixed)
I1-I3 hold for all three systems: planting a length cue in a benchmark that has none reproduces the text-saturated, image-indifferent behaviour of Med-CMR, which is causal evidence (an intervention on the benchmark, not a correlation) that a length cue of this strength is sufficient. I1 holds and I3 fails: the models learn the cue but keep using the image, so the cue is not what removes image use. I1 fails: the models do not pick up a planted surface cue, and the paper reports that. I4 is read as evidence that the model relies on the planted cue rather than on content. No threshold is changed after results are seen.

## Limits (stated in advance)
One seed per system; filler text is artificial and appears on all options (its amount depends on the rank); planting changes the benchmark in more than one way (artificial words); sufficiency of a planted cue does not show that the original Med-CMR cue is what drove its behaviour.

## Amendments log
(A0-A13.)
