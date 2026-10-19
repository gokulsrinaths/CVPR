# Amendment A38 to pre-registration v1: does the visual faithfulness judge check the image? Three judge sizes and judge-side counterfactuals

Appended 2026-10-07, before any run below. v1 and A0-A37 unchanged. Motivation: A37 found, with one reproduction judge (Qwen2.5-VL-7B), that the faithfulness score is the same whether the judge is shown the true image or a black one (3.08 against 3.03) and stays at 80% for a model that described another case's image. Two objections apply: the organisers' judge may be larger or better, and the A37 contrasts mix model-side and judge-side effects. A38 repeats the judge-side tests with three judge sizes and adds the cleanest test of whether a judge checks grounding at all: a trace from another case judged against this case's image.

## What has been seen (not predictions)
A37 (`analysis/a37_results.md`): with Qwen2.5-VL-7B the mean RVF_trace is 3.08 (true-image model, judge shown the true image), 3.03 (same traces, judge shown a black image), 0.30 (black-image model) and 2.45 (wrong-image model); 89.7% of the black-image traces say that the image is black or blank. No other judge size, no judge-side mismatch of traces and no text-only judging has been run.

## Judges and conditions (`scripts/a38_judge.py`, `slurm/48_a38_judge.sbatch`, hashed before the first submission)
Qwen2.5-VL-3B-, -7B- and -32B-Instruct with the scoring-suite VA rubric, greedy decoding (the organisers' exact judge size and prompt are not known; "Qwen-2.5-VL" is the only specification). The 175 open-ended held-out cases and the outputs of the released adapter of A37. Conditions: **C1** own trace, true image; **C2** own trace, black image; **C3** own trace, image of another case of the same modality (the A37 assignment); **C4** own trace, no image at all; **C5** the answer and trace of another case of the same modality (the A37 assignment), judged with this case's question and true image; **C6** the black-image model's trace, true image; **C7** the wrong-image model's trace, true image.

## Rules (per judge; paired bootstrap over the 175 cases, 2,000 resamples; RVF_trace)
- **K1 (the judge uses the identity of the image it is shown):** RVF(C1) minus RVF(C3) is at least 0.5 with the lower 95% bound at least 0.25.
- **K2 (the judge checks that a trace is about this case):** RVF(C1) minus RVF(C5) is at least 1.0 with the lower bound at least 0.5.
- **K3 (A37 M1 replicates):** RVF(C1) minus RVF(C6) is at least 1.0 with the lower bound at least 0.5.
Reported, not tested: C2 and C4 against C1, C7, VA_answer and VA_final.

## Reading rule (fixed)
- K1 and K2 fail for all three judges: the paper states that none of the three reproduction judges checks the image or the case of a trace, so the gate scores what the text says; the strongest form of the claim, still limited to Qwen2.5-VL-based reproductions.
- K1 or K2 holds for some judge sizes only: the paper states for which sizes, that the judge's grounding check depends on its size, and that the organisers' judge size decides which statement applies to the challenge.
- K1 and K2 hold for all: the paper states that the A37 sensitivity is specific to the 7B judge and removes the claim that the gate does not use the image.
- K3 is reported either way.
No run is repeated or tuned after seeing its result.

## Limits (stated in advance)
The organisers' prompts, judge size and decoding are unknown; one system and 175 cases; same-modality swaps may yield traces that are plausible for the image (a lower bound on how much a judge should detect); no GT judge; Qwen2.5-VL-32B is run in bf16 across the available GPU memory.

## Amendments log
(A0-A38.)
