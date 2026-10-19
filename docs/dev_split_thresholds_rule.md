# Rule for choosing the near-duplicate thresholds (tau_img, tau_cap) and building the dev split

Written 2026-10-02 BEFORE any image pair is reviewed. Implements `analysis/dev_split_protocol.md` section 3 with the practical
limits stated here. Read-only and hashed once uploaded. It uses only train-train pairs; test cases are never used to choose a threshold.

## Who reviews, and the honest limit
The "manual review" is done by the AI assistant (Claude) looking at side-by-side image montages, not by a human expert. It
judges only "same figure / same source image (including crops, rescaling, annotation overlays, or a screenshot of the same
scan)", never anything clinical. This is recorded in the paper as an automated near-duplicate audit. The protocol's 40-pair
sample per threshold is reduced to **10 pairs per distance bin** (seeded sample, excluding pairs that are already
byte-identical); with 10 pairs a bin "passes" only if **10 of 10** are judged the same (the protocol's 95% bar cannot be met
with one miss in 10).

## Bins and thresholds
- Image edges (E2): bins of the smaller of the two Hamming distances (pHash, dHash) = 1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13-14.
  tau_img = the upper edge of the last bin such that it and every earlier bin pass. If the first bin fails, tau_img = 0
  (only byte-identical images, E0).
- Caption edges (E3): bins of token-set Jaccard 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0 (pairs that also share an
  identical image are excluded from the sample). tau_cap = the lower edge of the lowest bin such that it and every higher bin pass;
  if the 0.9-1.0 bin fails, E3 is not used (E1, identical captions, remains).
- E0 (identical sha256), E1 (identical normalised caption) and E4 (>= 3 identical options and the same primary modality) are used
  as they are, without review. Reviewed sample ids, the judgements and the chosen thresholds are saved in
  `analysis/threshold_review.json` before the dev split is built.

## Split construction (deterministic)
1. Graph nodes: the 17,722 train cases. Edges: E0, E1, E2 (<= tau_img), E3 (>= tau_cap), E4. Components by union-find.
2. N(T): every pool case in a component containing at least one test case. R_test = pool minus N(T) is the retrieval corpus for
   test evaluation; N(T) is reported by edge type.
3. Dev split D: whole components drawn from the pool components that contain no test case, with a keyed hash
   (sha256 of seed 20260930 + the smallest case id in the component), stratified by question type x primary modality, until D holds
   about 15% of the pool cases. R_dev = pool minus D minus N(T).
4. Components above 50 cases are listed for inspection and kept whole. All manifests (case ids, component ids, counts, sha256)
   are written and made read-only before any model or retrieval run reads D.
