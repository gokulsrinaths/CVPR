# Amendment A3 to pre-registration v1: replacement readout for H1

Appended 2026-10-02. v1, A0 and A2 are unchanged. A1 stays reserved for the retrieval and rewrite predictions.

## What happened to J2 as pre-registered (job 22436141, completed, reported in full)
With the primary readout of `docs/gate_B_prespec.md` (teacher-forced letter read after the adapter's constant stub), the unadapted
Qwen3.5-9B scored 21.7% (true image) and 24.9% (black image): chance level (0.20). By the rule written in advance, **H1 is
NOT EVALUABLE** on that readout. The secondary readouts were also weak: `cue` 38.5% true / 36.9% black (loss 1.7 points,
exact McNemar p = 0.27) and `empty-think` 23.9% / 22.9% (loss 1.0 point, p = 0.30). Reading letters out of a teacher-forced
position measures the adapter's trained answer format, not the base model's ability, so the readout does not suit this model.
These secondary numbers were seen before this amendment; they motivated the change of readout and nothing else. No threshold below
differs from H1 as pre-registered.

## Replacement test H1'  (job name `gateB_h1gen_base_s0`)
- **Model and cases:** as J2 (4-bit Qwen3.5-9B, no adapter, detector/MAP/preprocessing unchanged; the 711 test MCQs; arms true
  image and black 224x224 image).
- **Prompt:** the frozen pipeline's system prompt, MAP note and user text, byte for byte, with the chat template's
  `enable_thinking=False`. No instruction is added (the system prompt already says "output only the option letter").
- **Decoding:** greedy, `max_new_tokens=16`, `repetition_penalty=1.0`. **Parsing:** the first standalone letter A-E in the generated
  text (regex `(?<![A-Za-z])([A-E])(?![A-Za-z])`); no letter = "unparsed", counted wrong.
- **Rule (unchanged from H1):** supported if accuracy(true) - accuracy(black) <= 5.0 points; **not evaluable** if accuracy(true) <
  0.60 or if more than 10% of generations are unparsed in either arm. Also reported: Wilson 95% intervals, exact McNemar (true vs
  black), the share of predictions equal to the shortest option (character rule of A0), accuracy on the 63 shortest-wrong cases.
- **If H1' is also not evaluable,** H1 is dropped as a confirmatory hypothesis and reported as untestable with this model; no
  third readout is tried for H1. Any second-system claim then rests on a different system under a new amendment.

## Amendments log
(A0, A2, A3.)
