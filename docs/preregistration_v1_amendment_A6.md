# Amendment A6 to pre-registration v1: length-equalised options (causal test of reliance on the length cue)

Appended 2026-10-03, before any rewrite exists. v1, A0-A5 unchanged. This is the "option-length rewrite" that A1 section 4 declared not run.

## Question
Does the adapter's accuracy fall when the length of the five options is equalised while each option's medical meaning is kept? If its picks come from the length cue, accuracy
should fall; the picture should then matter more. Until now only correlations were shown (`analysis/length_feasibility.md`, `gateBC_results.md`).

## What has already been seen (not predictions)
Baseline accuracy on TEST711 (96.06%) and DEV700 (97.0%), the black-image arms, the length statistics of the original options (gold is the shortest by words in 691/711 cases;
hedge words 0.38 per distractor vs 0.15 per gold option). Nothing about rewritten options exists yet.

## Rewriter (job `lr_rewrite_s0`)
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (local, bf16), used as a text-only model, greedy decoding. It never sees the question, the image or any label, so it cannot favour the
  correct option. Cases: the 711 test MCQs and the 700 DEV700 MCQs of H2 (same ids as `gateC_h2_dev_s0_22454795_*/groups.json`).
- **Prompt (fixed):** system `You are a careful medical editor.` User: `Below are the five options of a multiple-choice question. Rewrite each option so that all five
  have about {T} words (between {lo} and {hi} words), keeping each option's medical meaning exactly the same. Do not add new medical claims, do not remove key findings, do not
  change numbers, laterality or negations, and do not add or remove hedging words (for example likely, may, suggests, typically). Do not make any option more or less plausible
  than it was. Return only a JSON object with keys A, B, C, D, E.` followed by the five original options as `A: text` lines. {T} = the median word count of the five original
  options (rounded), {lo} = round(0.9 T), {hi} = round(1.1 T).
- **Second attempt:** a question whose first output fails the parse or the filters below is re-run once with sampling (temperature 0.7, seeds fixed by case id); the output is
  accepted only if it passes the filters. A question that still fails is **excluded from the equalised analysis** (label-free exclusion; the count is reported).

## Label-free filters (per question; all must hold)
1. JSON with exactly the keys A-E, each a non-empty string. 2. max/min word count of the rewritten options <= 1.30. 3. For every option, the multiset of numbers (digits and decimals)
equals that of the original. 4. For every option, the set of negation words {no, not, without, absent, non, never, none, neither, nor, lack, lacks, lacking, negative} equals the original's.
5. For every option, the set of {left, right, bilateral, unilateral} equals the original's.

## Gates (all must hold, or the experiment is reported NOT EVALUABLE and no model is run on the rewrites)
- **G1 (manipulation works), on TEST711:** at least 60% of questions pass the filters; the median max/min word ratio of the passing questions is <= 1.20; the shortest-option rule
  (A0 character rule) applied to the rewritten options of passing questions has accuracy <= 0.40.
- **G2 (meaning preserved), hand check:** 50 passing TEST711 questions drawn with `random.Random(20261003)`; the assistant (not a human expert) compares original and rewritten options, knowing which is
  correct, and judges each question "preserved" only if every option keeps its medical claim and the correct option is still correct and the wrong ones still wrong. At least **45 of 50** must be preserved.
  The judged ids and verdicts are saved in `analysis/lr_handcheck.json` before any model sees the rewrites.

## Evaluation (job `lr_eval_s0`; only if G1 and G2 pass)
The frozen adapter pipeline, teacher-forced `direct` readout (cue secondary), on the passing questions of TEST711 and of DEV700, with the rewritten option texts (labels unchanged, same
order). Arms: `eq` (true image) and `eq_black` (black image). Originals: the Gate A base arm (test) and the H2 base arm (dev), restricted to the same passing questions.

## Predictions (each falsifiable)
- **L1 (reliance on the cue), TEST711 passing set:** accuracy(original) - accuracy(eq) >= 5.0 points with exact McNemar p < 0.05 => reliance **supported**. A drop <= 2.0 points => **no reliance**
  (the shortcut story is falsified for this model). Between 2.0 and 5.0, or p >= 0.05, => inconclusive.
- **L2 (replication), DEV700 passing set:** the same rule as L1 (labelled in-sample suspected).
- **L3 (the picture matters more once the cue is gone), TEST711 passing set:** accuracy(eq) - accuracy(eq_black) >= 5.0 points. Falsified below 5.0; **not evaluable** if accuracy(eq) < 0.40.
- **Reported without prediction:** the cue readout; the shortest-option rule's accuracy before and after; hedge-word counts per option before and after (a residual cue); accuracy on
  the cases where the shortest original option was wrong; all excluded counts.

## Limits stated in advance
One 7B rewriter and one adapter; the assistant's hand check is not an expert review; the rewrite keeps hedge words, so a residual cue remains and a drop measures reliance on length together
with whatever the rewriter changed, not length alone; this is a causal test of the length cue, not of every text cue.

## Amendments log
(A0, A1, A2, A3, A4, A5, A6.)
