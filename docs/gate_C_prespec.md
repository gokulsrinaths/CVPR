# Gate C pre-specification: H2 (black-image replication on the dev split)

Written 2026-10-02 BEFORE submission. Implements H2 of `docs/preregistration_v1.md` exactly; nothing here changes it.

## Context fixed by earlier results
Gate B J1 (job 22436140) flagged the development pool as **in-sample suspected** through the dev-protocol accuracy trigger (POOL700
accuracy 98.14%, Wilson lower bound 96.85% > 96.06%), while the confidence-distribution rule was not met. By the frozen
consequence (b), the dev split `data/train_pool/split_v1/dev_ids.json` **must not calibrate any threshold or measure any method
gain** until the owner decides. H2 is not a calibration or a method gain: v1 states it is "reported even if the validity gate says the
dev split is in-sample". Its result is labelled **in-sample suspected** wherever it appears.

## Job `gateC_h2_dev_s0` (adapter ON)
- **Cases:** DEV700 = 700 MCQs drawn with `random.Random(20261002).sample` from the MCQ ids of `split_v1/dev_ids.json` (ids sorted
  first). Gold = official train answers (used only in the summary).
- **Arms:** `base` (true image) and `black` (black 224x224). **Readouts:** `direct` (primary, as Gate A) and `cue`.
- **H2 (as pre-registered):** supported if accuracy(true) - accuracy(black) <= 5.0 percentage points on the primary readout;
  falsified otherwise. Also reported: Wilson 95% intervals, exact McNemar, the share of predictions equal to the shortest option
  (character rule of A0), accuracy on the cases in the sample where the shortest option is wrong.
- **Guards:** the job stops if the adapter does not attach (exit 5). No anchor group (the harness was verified against Gate A in
  J1, 20/20).
- Outputs, registry and hashes as in Gate B. No re-run except after a crash (resume).
