# Amendment A0 to pre-registration v1 (definition fix only; A1 stays reserved for retrieval and rewrite predictions)

Appended 2026-10-02, after `docs/preregistration_v1.md` (sha256 `f53a17c958424b01eb0fdc471bdc9cd7b6815fec865bc5a6eea790184688e1ed`,
frozen 2026-10-02T16:48:50Z). The v1 file is unchanged. This amendment is its own read-only file.

## What was wrong
v1 section 2 defines "shortest option" by whitespace-separated words, but H3's interval (88% to 94%) was anchored on the
648/711 = 91.1% rule from `scripts/phase2c_length_artifact.py`, which picks `min(options, key=(character length, letter))`: shortest
by **characters**, ties to the alphabetically first letter. The two are different statistics.

## How it was found, and what was seen first
After v1 was frozen, `scripts/length_feasibility.py` was run on the 711 test MCQs (`analysis/length_feasibility.json`). It showed the
word-based rule behaves differently: 614 cases have a unique shortest option by words, and the gold option is that option in 595
(96.9%); 97 cases are tied. So the word-based statistics on the test set were seen before this amendment was written. No threshold
or interval in v1 was moved because of them.

## Amendment
1. "Shortest option" everywhere in v1 (section 2, H3, H5, the 63-case subset) means the **character-length** rule of
   `phase2c_length_artifact.py`, with its tie rule (alphabetically first letter).
2. H3 is evaluated with exactly that rule on Pool MCQs. The interval 88% to 94% is unchanged.
3. The word-based rule is reported as a secondary, exploratory statistic, with its unique-shortest accuracy and tie count shown
   separately.

## Amendments log
(A0 only.)
