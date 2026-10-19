# Amendment A1 to pre-registration v1: resolution of the dev-split decision, retrieval design, predictions

Appended 2026-10-02, before any retrieval embedding, retrieval run or retrieval arm exists. v1, A0, A2, A3, A4 are unchanged. It supersedes the
unfrozen proposal `analysis/A1_draft_for_owner.md`.

## 1. The decision (made by the assistant, flagged for the owner's ratification, reversible)
The validity gate (`analysis/gateBC_results.md`) left the development pool "in-sample suspected", and the frozen consequence (b) reserved the
choice of how to proceed for the owner (`analysis/owner_decision_dev_split.md`). The owner delegated ("Do whatever is necessary for this paper")
and set the goal of completing the process, so the assistant took **option 1**: the low-margin threshold tau_m is the **15th percentile of
the margins (direct readout) of the 700 unlabeled validation MCQs (VAL700 of job 22436140)**, cases the adapter cannot have trained on. No labels,
no in-sample case and no test case is used to set it.
- Deviation, stated plainly: rule 3 says the threshold is chosen "only on development data". The official validation split is the challenge's own
  development/validation data but is not part of the authorised 16,836-case pool named in rule 2, so this is a wording change, not a change of the
  rule's intent (no tuning on test). **If the owner rejects it,** all results below are reported as exploratory descriptive audits and nothing else changes.
- Everything else of rule 1-5 stands: test frozen, four views, the 7 image-necessary cases qualitative only, frozen artifacts untouched.
- The dev split v1 plays **no role** in the retrieval experiment (it was used only for H2). Encoder, k and prompt are fixed below and are
  **not selected on any labelled data.**

## 2. Design (fixed now; no tuning)
- **Encoder:** `flaviagiammarino/pubmed-clip-vit-base-patch32` (MIT, public), image embedding = `visual_projection(vision_model(pixel_values).pooler_output)`,
  L2-normalised, 512-d, original RGB image through the model's own processor. One encoder only, none compared.
- **Retrieval:** cosine similarity of image embeddings, within the same `primary_modality`, MCQ corpus cases only, top **k = 3** (a single
  value, not a grid). Query = the test case's own image embedding.
- **Corpora:** R_test = the pool minus the 62 same-source cases N(T) (`split_v1/r_test_ids.json`), MCQ only. R_full = the pool minus the 886 test
  ids (so including N(T)), MCQ only (leakage quantification only). A control draws 3 random same-modality cases from R_test (seeded by case id).
- **Injected text:** for each neighbour the question (first 300 characters) and the text of its verified correct option, as a block placed before
  the case's own question: `Reference cases with visually similar images (verified answers, for context):` then
  `(n) Q: ... / Verified answer: ...` per neighbour, a blank line, `Now the case to answer:`, then the original question text and options.
  Never the `visual_description` caption, never any test case, never the neighbour's option letter.
- **Readout:** the same teacher-forced `direct` readout (and `cue` secondary) as Gate A on the unmodified adapter pipeline. M0 = the Gate A base arm.
- **Arms on all 711 test MCQs:** M1 (R_test neighbours, true image); M1c (random same-modality neighbours, true image); M1leak (R_full neighbours, true
  image); M1black (R_test neighbours retrieved with the true image, black image shown to the model). M2 (gated) is computed at analysis time:
  use M1's answer where the M0 margin < tau_m, else M0's. Guard: the first 20 cases are also run without context and must agree with Gate A on at least 19.
- **Views (rule 3):** all 711; the 63 cases where the shortest option (A0 rule) is wrong; the low-margin subset (M0 margin < tau_m); the black-image control
  (M1black against the Gate A black arm).

## 3. Predictions (each falsifiable; v1's H4-H6 restated under the above)
- **H4:** net gain (fixed minus broken) of M1 over M0 is at most +4 on the 711. Falsified by +8 or more with exact McNemar p < 0.05.
- **H4c (control):** |net change| of M1c relative to M0 is at most 4. A larger change means a prompt-length effect, not retrieval.
- **H5:** the net gain of M1 on the 63 shortest-wrong cases is at most 0. Falsified if positive.
- **H6:** tau_m (from VAL700) flags low-margin test cases that contain at least 60% of the 28 test errors. Falsified below 60%.
- **H7 (gated retrieval):** net gain of M2 over M0 is at most +4 on the 711 (falsified by +8 or more, exact McNemar p < 0.05).
- **Reported without prediction:** M1leak; M1black against the black arm; the share of the 57 same-source test cases whose partner appears in the
  top-3 of M1leak (a check that the index finds same-source cases); retrieval diagnostics (same-modality rate, mean similarity).

## 4. Declared not run
Retrieve-to-refute, any option-length rewrite perturbation, and re-running option-order averaging (already observed on the test set in Gate A,
exploratory) are **not** run here. They are future work or need their own amendment before running.

## Amendments log
(A0, A1, A2, A3, A4.)
