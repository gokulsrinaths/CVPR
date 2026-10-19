# Gate B pre-specification: development-pool validity check (J1) and base-model image-dependence audit (J2)

Written 2026-10-02 BEFORE either job is submitted. Rules in `docs/evaluation_rules_frozen.md` and
`docs/preregistration_v1.md` (+ amendments A0, A2) apply. This file is made read-only and hashed before submission.
No threshold below was chosen after seeing a result of these jobs. Both jobs are teacher-forced forward passes (no sampling).

## Shared definitions
- **Readout.** The 5 letter log-probabilities, renormalised with a softmax over A-E. `direct` = read right after
  `<stub>\n</think>\n\n`; `cue` = read after `The correct option is`, both exactly as in Gate A (`scripts/mvqa_diag.py`). **Primary
  readout: `direct`.** The stub is the adapter's own constant MCQ reasoning (see `analysis/error_analysis_report.md`).
- **Margin m** = top-1 probability minus top-2 probability of that distribution (the Gate A definition).
- **Confident fraction c** = share of cases with m >= 0.90 (0.90 is the Gate A flag value, used here only as a convention).
- Pipeline, prompts, preprocessing, modality detector and image loading are the frozen baseline's, unmodified.

## J1: validity check (adapter ON), job name `gateB_validity_s0`
**Question:** is the development pool in-sample for the adapter, i.e. is the adapter more confident on pool cases than on cases it
cannot have trained on? The validation split (answers never released) is the reference for "unseen" and was built by the
organisers with the same stratification as train.
- **Groups.** POOL700 = 700 MCQs drawn with `random.Random(20261002).sample` from the authorised pool (official train MCQs minus
  the 886 test ids, ids sorted first). VAL700 = 700 MCQs drawn the same way from the 2,057 official validation MCQs (no labels
  used). ANCHOR20 = the first 20 test MCQs in `cases.json` order. TEST711 = the Gate A base-arm rows already on disk (not re-run).
- **Arm:** `base` only (true image). No greedy generation.
- **Harness guard.** ANCHOR20 `direct` argmax must equal Gate A's for at least 19 of 20, otherwise the job stops with exit 3.
- **Rule (applied separately to POOL700-vs-VAL700 and to TEST711-vs-VAL700).** The group is **in-sample suspected** if ALL hold:
  (i) two-sample Kolmogorov-Smirnov p < 0.01 on m; (ii) median m of the group exceeds median m of VAL700; (iii) c(group) - c(VAL700)
  >= 0.05. Otherwise **no evidence of memorisation** (a statement about this resolution only: with 700 per group a 5-point
  difference in c is detectable with roughly 90% power at c near 0.9).
- **Second trigger from the dev protocol (POOL700 only).** Accuracy of the `direct` argmax against the official answers: in-sample
  suspected if accuracy >= 0.99 or the Wilson 95% lower bound exceeds 0.9606.
- **Consequences, fixed now.** (a) POOL not suspected: a pool-derived dev split may be used for calibration, with the stated
  resolution caveat. (b) POOL suspected: the dev split must not calibrate any threshold or measure any method gain; the owner
  decides between the dev-protocol options. (c) TEST711 suspected: the headline baseline numbers on the 711 test MCQs are in doubt;
  stop and report to the owner before any further use of the test set.

## J2: base-model image-dependence audit (adapter OFF), job name `gateB_h1_base_s0`
**Question (H1):** does the unadapted Qwen3.5-9B also keep its accuracy when the image is blanked?
- **Model:** the same 4-bit base weights, no LoRA (`MEDREASON_ADAPTER_PATH` pointed at a non-existent directory; the job stops if
  an adapter attaches). Detector, MAP prompt and preprocessing unchanged.
- **Cases:** the 711 test MCQs. **Arms:** `base` (true image) and `black` (the code's black 224x224 fallback).
- **Readouts:** `direct` (stub-forced, primary: identical forcing for both systems, so the comparison isolates image use, with the
  caveat that the stub is the adapter's text) and `cue`; plus `empty` (secondary): the same forward with an empty thinking block
  (`<think>\n\n</think>\n\n`), the closest thing to the base model's native non-thinking format.
- **H1 (as pre-registered):** supported if accuracy(true image) - accuracy(black image) <= 5.0 percentage points on the primary
  readout. **Not evaluable** if accuracy(true image) < 0.60 on the primary readout (the model is too weak for the comparison); the
  `empty` readout is then reported and a replacement test is a new amendment.
- **Always reported:** both accuracies with Wilson 95% intervals, the paired exact McNemar test (true vs black), the share of
  predictions equal to the shortest option (character rule of amendment A0) in each arm, and accuracy on the 63 cases where the
  shortest option is wrong.

## Not done, by rule
No threshold, readout or case selection is changed after a result. A job is re-run only after a crash (resume the same run
directory). Every outcome is reported, whichever way it goes. Test labels are used only inside the final summaries.

## Outputs
`experiments/pilot/<job>_<slurmid>_<timestamp>/{config.json,env.json,groups.json,forward_rows.jsonl,errors.jsonl,summary.json}`,
registered in `experiments/job_registry.csv`. Scripts: `scripts/audit_forward.py`, `scripts/audit_summarize.py` (hashes in config).
