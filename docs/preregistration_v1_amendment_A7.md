# Amendment A7 to pre-registration v1: second (and last) design of the length-equalising rewrite

Appended 2026-10-03, before any A7 rewrite exists. v1, A0-A6 unchanged. A6's protocol, predictions L1-L3 and the evaluation job stand; only the rewriting procedure and gate G1 change.

## Why a second design
Attempt 1 (A6, job 22564998) failed gate G1: 17.6% of test questions passed the filters, the passing questions were already more balanced than average (original shortest-option accuracy 71.2%
against 91.1% overall), and equal word counts left character lengths unequal (shortest-option rule 53.6% after rewriting). Report: `analysis/lr_rewrite_attempt1_results.md`. Those numbers were seen
before this amendment and motivated the three changes below. **This is the last rewrite design: if G1 fails again the length-equalisation experiment is reported NOT EVALUABLE and not retried;
a better result would need human editing of options, which is out of scope.**

## Changes
1. **One option at a time.** Each of the 5 options of a question is rewritten in its own request (the rewriter sees only that option's text: no question, no other option, no image, no label), so it
   cannot favour the correct option and all five receive the same treatment (no option is left unchanged).
2. **Feedback loop.** Up to 6 tries per option: try 1 greedy; tries 2-6 sampled (temperature 0.7, seeded per batch from the case ids) as a follow-up turn telling the rewriter its previous text's character
   and word counts and the band it must meet. The first try that satisfies the acceptance rule below is kept; if none does, the option is marked failed.
3. **Character and word targets from the question itself.** T_c = the median character count of the five original options, T_w = their median word count. **Acceptance of an option:** characters in
   [min(floor(0.92 T_c), T_c - 4), max(ceil(1.08 T_c), T_c + 4)]; words in [max(1, floor(0.85 T_w)), max(ceil(1.15 T_w), T_w + 1)]; the multiset of numbers, the set of negation words
   {no, not, without, absent, non, never, none, neither, nor, lack, lacks, lacking, negative}, the set of {left, right, bilateral, unilateral}, and the set of hedging words
   {generally, typically, likely, may, might, possibly, suggests, suggesting, suggest, usually, often, could, probable, probably, potentially, indicating, consistent, appears, apparent, suggestive}
   all equal to the original option's; one line of text. A **question passes** only if all five options are accepted.
4. **Prompt (fixed):** system `You are a careful medical editor.` User: `Rewrite the following answer option from a medical multiple-choice question in different words so that it has between {c_lo} and {c_hi}
   characters (about {w_lo} to {w_hi} words), keeping its medical meaning exactly the same. Do not add new medical claims, do not remove key findings, do not change numbers, laterality or negations, and do
   not add or remove hedging words (for example likely, may, suggests, typically). Return only the rewritten option text.` followed by a blank line and `Option: {text}`. Feedback turn: `That rewrite has {c}
   characters and {w} words. It must have between {c_lo} and {c_hi} characters and between {w_lo} and {w_hi} words, with the same meaning. Return only the corrected option text.`
5. **Model, cases:** Qwen2.5-VL-7B-Instruct text-only (as A6); TEST711 and DEV700 (same ids).

## Gates (all must hold on TEST711, else NOT EVALUABLE and no model is run on the rewrites)
- **G1a:** at least 60% of questions pass. **G1b (manipulation works):** among passing questions, the shortest-option rule (A0 character rule) applied to the rewritten options has accuracy <= 0.40.
  **G1c (no selection bias):** among passing questions, the same rule applied to the ORIGINAL options has accuracy >= 0.82 (within about 10 points of the 91.1% overall).
- **G2 (meaning preserved):** the hand check of A6 (50 passing TEST711 questions, `random.Random(20261003)`, judged by the assistant knowing the correct option, at least 45 of 50 preserved), saved in
  `analysis/lr_handcheck.json` before any model sees the rewrites.

## Evaluation and predictions
Exactly A6: job `lr_eval_s0` on the passing questions (arms eq and eq_black), predictions L1 (reliance: drop >= 5.0 points with exact McNemar p < 0.05; no reliance: drop <= 2.0), L2 (DEV700 replication),
L3 (image effect on equalised text >= 5.0 points, not evaluable if accuracy < 0.40). Reported without prediction: the cue readout, shortest-option rule before and after, hedge counts, accuracy on cases where
the original shortest option was wrong, all exclusion counts.

## Limits stated in advance
One 7B rewriter; the assistant's hand check is not an expert review; the passing subset may still differ from the full benchmark in ways G1c does not capture; hedge words are kept (a residual cue), so a drop
measures reliance on length together with whatever paraphrasing changes.

## Amendments log
(A0, A1, A2, A3, A4, A5, A6, A7.)
