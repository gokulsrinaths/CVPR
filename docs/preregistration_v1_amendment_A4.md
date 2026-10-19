# Amendment A4 to pre-registration v1: second system (Qwen2.5-VL-7B-Instruct) image-dependence audit

Appended 2026-10-02, before the model has been run on any case. v1, A0, A2, A3 are unchanged; A1 stays reserved.

## Why
H1 could not be evaluated for the unadapted Qwen3.5-9B (A3). Generality across systems is a claim of the paper, so it is tested on a
different, public system that follows a letter-only instruction. Lesson from A3 (my design flaw): the generation budget and the
instruction must suit the model, so both are fixed here in advance and checked in a CPU self-test before any GPU time.

## System S2 (job `gateC_s2_qwen25vl_s0`)
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, public), bf16, sdpa attention, one H100, greedy decoding, `max_new_tokens=8`.
  Weights downloaded to `models/qwen25vl_7b_instruct`; the job runs offline.
- **Prompt:** system message `You are a medical imaging expert.`; user message = the image, then the pipeline's MCQ text (the question,
  a blank line, `Options:`, one line `  X: text` per option) followed by a blank line and
  `Answer with only the letter (A, B, C, D or E) of the correct option.` No thinking mode, no modality detector, no MAP note.
- **Images:** the original RGB image downscaled with LANCZOS so the longest side is at most 896 px (never upscaled). The black arm is a black 224x224 image.
- **Parsing:** the first standalone letter A-E in the generated text (regex `(?<![A-Za-z])([A-E])(?![A-Za-z])`); none = unparsed, counted wrong.
- **Cases:** TEST711 (the 711 test MCQs, the audit's fixed evaluation set) and DEV700 (the same 700 dev MCQs as H2, ids recomputed with
  seed 20261002 and checked against H2's `groups.json`). Arms: true image, black image. Labels are used only in the summary.
  Qwen2.5-VL is unrelated to the MedReason adapter, so the in-sample concern of the dev split does not apply to it; this is stated, not tested.

## Predictions (each falsifiable)
- **S2-a (evaluable):** accuracy with the true image >= 0.60 and unparsed rate <= 10% in both arms, on TEST711. Otherwise the whole
  test is **not evaluable**, is reported as such, and is not retried with another prompt or budget (a further system needs A5).
- **S2-b (image dependence, the H1 statement for S2):** accuracy(true) - accuracy(black) <= 5.0 percentage points on TEST711. Falsified above 5.0.
  The same statement is evaluated on DEV700 (replication) and reported separately.
- **S2-c (length shortcut):** on TEST711 with the true image, the share of predictions equal to the shortest option (character rule of A0)
  is >= 0.50. Falsified below 0.50.
- **Always reported:** Wilson 95% intervals, exact McNemar (true vs black), accuracy on the 63 shortest-wrong cases (TEST711) and the
  shortest-wrong cases in DEV700, and the unparsed rates.

## Amendments log
(A0, A2, A3, A4.)
