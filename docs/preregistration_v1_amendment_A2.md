# Amendment A2 to pre-registration v1 (procedure clarification)

Appended 2026-10-02, after `docs/preregistration_v1.md` (sha256 `f53a17c9…`) and `docs/preregistration_v1_amendment_A0.md`
(sha256 `cea18e84…`). Neither is changed. A1 stays reserved for the retrieval and rewrite predictions.

## Clarification
v1 section 5 says train-side statistics are computed only after the train zip is on the cluster and a sha256 manifest exists. The
zip was instead obtained as follows, and the data freeze it was meant to give is in place:
- The official zips were downloaded by the owner and their MD5 values matched Synapse (train `fd3fb586814b08869d7d856b7d706be2`,
  validation `e470b318793e7562fed7ef6579120663`).
- The two official JSON files were extracted from those zips, copied to the cluster, and hashed there (train
  `8624534b53a36caae19027ac7b892137180eb6b6a76dbb33b68e3ec3ea96ecf1`, validation
  `9c03850d08a70c7e225813559804136fe0ef00d5fc5c49bb63e479c74ce762d2`) in `data/train_pool/SHA256_MANIFEST.txt`, read-only.
- The 20,254 official images were taken from the public Med-CMR zip and are byte-identical (sha256) to the images in the official
  zips; the 400 reserved leaderboard images were never extracted. `data/train_pool/IMAGE_MANIFEST.json` is read-only.

For statistics that need only the JSON files (H3 and the non-image acceptance checks), this freeze satisfies the purpose of the
rule. The retention of the original zip file on the cluster is not required.

## What had been computed before this amendment (disclosure)
No answer-length, accuracy or shortcut statistic on the train data. Only: counts, field names and modality/task counts from the
JSON (before v1), and label-free image-hash statistics (exact-duplicate images between test and pool, in
`analysis/test_pool_image_twins.json`), which use no answers.

## Amendments log
(A0, A2.)
