# Pre-registration v1: vision-necessity audit

Written 2026-10-02, before any run listed in section 3 exists. Changes are dated amendments appended at the bottom; nothing above
the amendments section is ever edited. This file is made read-only and hashed when it is created (`analysis/preregistration_v1.sha256`).
It sits under `docs/evaluation_rules_frozen.md`, whose five rules still apply.

## 1. What has already been observed (NOT predictions)

All of this was seen on the frozen 711-case test set (Gate A and the length analysis), so it motivated the hypotheses below and
cannot confirm them. Confirmation must come from data or runs that did not exist when these numbers were seen.

- Baseline 683/711. Black-image arm 95.2%. Shortest-option rule 648/711 (91.1%); the gold option is the (non-strictly) shortest
  in 655/711; the longest option is correct 0/711. Median gold length / mean distractor length = 0.579.
- Among the 63 cases where the shortest option is wrong, the baseline is right on 51 and wrong on 12; among the 648 where it is
  right, wrong on 16.
- Permutation averaging 692/711, mostly via cases where the shortest option is already correct. 7 image-necessary cases.
- The official train zip (17,722 cases) was downloaded and only its metadata, counts and field names were read. No answer-length,
  accuracy or shortcut statistic has been computed on it. That stays true until its sha256 manifest exists (section 5).

## 2. Definitions fixed now

- **Image-necessary case:** correct with the true image AND incorrect under both counterfactual images (black, mismatched), using
  the same two arms as Gate A. Never a standalone statistical claim.
- **Shortest option:** fewest whitespace-separated words after removing the "X) " label; ties are reported separately and are
  never counted as "shortest correct" in a headline number.
- **Dev MCQs:** MCQ cases of the dev split built by `analysis/dev_split_protocol.md`. **Pool MCQs:** the MCQ cases of the
  official train file (14,398) after removing the 711 test MCQs, before any split.
- **Confirmatory family:** H1-H6, Holm-adjusted together at alpha 0.05. Everything else is exploratory and labelled so.

## 3. Hypotheses (directional; each lists the result that falsifies it)

- **H1 (generality across systems).** The base Qwen3.5-9B with no adapter, same prompt and greedy decoding, loses at most 5
  accuracy points when the image is replaced by a black image, on the 711 test MCQs. *Falsified if the loss exceeds 5 points.*
- **H2 (replication on other data).** The adapter system loses at most 5 points under the black image on dev MCQs. *Falsified if
  the loss exceeds 5 points.* Reported even if the validity gate says the dev split is in-sample.
- **H3 (the length rule is not specific to the held-out slice).** The shortest-option rule's accuracy on Pool MCQs lies within
  88% to 94%. *Falsified outside that interval.* This needs no GPU and is computed once, after the manifest in section 5.
- **H4 (plain retrieval).** Plain top-k retrieval, k chosen on dev, gives a net gain (fixed minus broken) of at most +4 on the
  711 test MCQs. *Falsified by a net gain of +8 or more with exact McNemar p < 0.05.*
- **H5 (where gains come from).** Any retrieval gain is concentrated where the shortest option is already right: the net gain on
  the 63-case shortest-wrong subset is at most 0. *Falsified if that net gain is positive.*
- **H6 (uncertainty gate transfers).** The margin threshold chosen on dev under the rule in the dev protocol (flag at most 15% of
  dev MCQs, capture the largest share of dev errors, infeasible below 80%) captures at least 60% of the 28 test errors.
  *Falsified below 60%.*

The retrieve-to-refute method and any option-length-rewrite perturbation have no predictions yet. **No retrieval run and no
rewrite run happens before Amendment A1 records their predictions.**

## 4. Analysis rules

- Paired comparisons use exact McNemar; proportions use Wilson 95% intervals. All outcomes are reported, including failures.
- Each method is run once on the 711 test MCQs with its dev-selected configuration, then reported on the four views in rule 3.
- The test set is read only by the final evaluation script, whose hash and run time are appended to `TEST_ACCESS_LOG.md`.
- If the validity gate flags the dev split as in-sample, H2 and H6 are still reported, labelled as in-sample, and no threshold is
  taken from that split without the owner's decision.
- Proxy judges are never described as official. Results on open-ended GT/VA carry that label.

## 5. Order of events that keeps this valid

1. This file is created, made read-only, and hashed.
2. The train zip arrives on the cluster, is verified, and a sha256 manifest is written and made read-only.
3. Only then are H3 and any train-side statistic computed.
4. Amendment A1 (retrieval and rewrite predictions) is written before the first dev retrieval run.

## Amendments
(none)
