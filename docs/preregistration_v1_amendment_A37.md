# Amendment A37 to pre-registration v1: does the challenge's faithfulness-gated visual score respond to the image?

Appended 2026-10-06, before any run below. v1 and A0-A36 unchanged. Motivation: the MedReason 2026 challenge scores open-ended answers with a visual-accuracy judge whose result is gated by the faithfulness of the reasoning trace (RVF_trace: does the reasoning stay grounded in visible image evidence), and the released system's own paper reports that faithfulness rises while answer correctness barely moves. If an answer produced without seeing the image receives the same faithfulness score, the gate rewards well-formed visual-sounding text, not image use. A37 asks this of the released adapter.

## What has been seen (not predictions)
The released adapter (Qwen3.5-9B, 4-bit, rank-16 LoRA; `models/adapter`) was run on the 886 held-out cases with the true images (`data/holdout886/run_output/`, MCQ 683/711); its open-ended answers were scored only with the heuristic proxy judge of the scoring suite (not the official judges). Its traces contain statements such as "The image is an axial CT scan..." and mention image labels. No open-ended run with a black or wrong image and no judge-model scoring of these outputs exists. The paper A8 results show that its multiple-choice answers do not change under counterfactual images; nothing is known about the open-ended traces.

## Runs
Inputs (`scripts/a37_build.py`): the 175 open-ended cases of the held-out slice with the image replaced by (true) the original, (black) an all-black image of the same size, (wrong) the image of another open case of the same modality (seeded derangement within modality, `data/holdout886_open_wrong/assignment.json`). Inference: the released container code unchanged (`code/process.py`, `MEDREASON_SYSTEM=custom`, the released adapter, detector and decoding), one job per arm (`slurm/46_a37_infer.sbatch`), A30 GPUs. Judging (`scripts/a37_judge.py`, `slurm/47_a37_judge.sbatch`, hashed before the first submission): Qwen2.5-VL-7B-Instruct, the VA rubric of the scoring suite (VA_answer and RVF_trace on 0-4, VA_final by the challenge's gate), greedy decoding; **the judge sees the true image of the case** for the three model arms, and, as a judge-side control, a black image for the outputs of the true-image arm. This is a reproduction: the organisers' judge prompts, model size and Llama-based GT judge are not available, so GT is not scored.

## Rules (paired bootstrap over the 175 cases, 2,000 resamples)
- **M1 (the faithfulness score responds to removing the image):** mean RVF_trace of the true-image arm minus that of the black-image arm is at least 1.0 point with the lower 95% bound at least 0.5.
- **M2 (the gated score responds):** the same for VA_final: difference at least 0.5, lower bound at least 0.25.
- **M3 (it responds to the wrong image):** RVF_trace true-image arm minus wrong-image arm at least 1.0, lower bound at least 0.5.
- **J1 (the judge uses the image it is shown):** for the true-image arm, RVF_trace with the true image minus with a black image at the judge is at least 0.5, lower bound at least 0.25.
Reported, not tested: VA_answer, shares of traces that mention a black or blank image, trace length, share of answers identical to those of the true-image arm.

## Reading rule (fixed)
- M1 and M2 fail: the paper states that, under this reproduction judge, the faithfulness gate does not separate outputs produced with the image from outputs produced without it, so the visual score does not measure image use (the strongest claim; it holds for the reproduction judge, not for the organisers' judge).
- M1 holds, M3 fails: it senses the presence of an image but not whether it is the right one.
- M1, M2 and M3 hold: the score responds to image content; the paper reports it as evidence that the metric is sensitive, and that the saturation lies in the multiple-choice track.
- J1 fails: in addition, the reproduction judge does not use the image, and every statement above is limited to that judge.
No run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One released system; 175 cases; a reproduction judge of a different prompt (and possibly a different size) from the organisers'; no GT scoring; the judge may be affected by the verbosity of the traces; wrong-image arm uses the same modality only; decoding of the container includes a forced-conclusion step that may behave differently with a black image.

## Amendments log
(A0-A37.)
