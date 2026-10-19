#!/usr/bin/env python
"""
Phase 2c (CPU only): quantify the option-length artifact and correct the McNemar table.
Reads analysis/mcq_all711_features.csv. Writes analysis/length_artifact.json.
"""
import collections
import csv
import json
import math
import re

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
AN = f"{ROOT}/analysis"
rows = list(csv.DictReader(open(f"{AN}/mcq_all711_features.csv")))


def parse_opts(s):
    out = {}
    for part in s.split(" || "):
        m = re.match(r"^([A-E]): (.*)$", part, re.S)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def rank_by_len(o, k):  # 1 = shortest ... 5 = longest (ties share the lower rank)
    return 1 + sum(len(o[j]) < len(o[k]) for j in o)


gt_rank = collections.Counter()
pred_rank = collections.Counter()
tie_cases = 0
tab = collections.Counter()  # (model_correct, shortest_correct)
model_wrong_when_shortest_right = []
model_right_when_shortest_wrong = []
for r in rows:
    o = parse_opts(r["options"])
    g, p = r["ground_truth"], r["parsed_prediction"]
    lens = sorted(len(v) for v in o.values())
    if lens[0] == lens[1]:
        tie_cases += 1
    s = min(o, key=lambda k: (len(o[k]), k))
    gt_rank[rank_by_len(o, g)] += 1
    pred_rank[rank_by_len(o, p)] += 1
    mc, sc = int(p == g), int(s == g)
    tab[(mc, sc)] += 1
    if mc == 0 and sc == 1:
        model_wrong_when_shortest_right.append(r["case_id"])
    if mc == 1 and sc == 0:
        model_right_when_shortest_wrong.append(r["case_id"])

# length statistics of the gold option vs the mean distractor
ratio = []
for r in rows:
    o = parse_opts(r["options"])
    g = r["ground_truth"]
    d = [len(v) for k, v in o.items() if k != g]
    ratio.append(len(o[g]) / (sum(d) / len(d)))
ratio.sort()
median_ratio = ratio[len(ratio) // 2]


def mcnemar_p(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


detect = {}
for b in (0, 1, 2, 3, 5, 8, 12):
    need = next((f for f in range(b + 1, 29) if mcnemar_p(f, b) < 0.05), None)
    detect[f"broken={b}"] = {"min_fixes_needed_for_significant_IMPROVEMENT": need,
                             "net_gain_pp": None if need is None else round(100 * (need - b) / 711, 2)}

out = {
    "gold_option_length_rank_distribution (1=shortest..5=longest)": dict(sorted(gt_rank.items())),
    "model_pick_length_rank_distribution (1=shortest..5=longest)": dict(sorted(pred_rank.items())),
    "cases_with_tied_shortest_length": tie_cases,
    "median_ratio_gold_len_over_mean_distractor_len": round(median_ratio, 3),
    "model_vs_shortest_heuristic_2x2": {
        "model_correct_and_shortest_correct": tab[(1, 1)],
        "model_correct_and_shortest_WRONG": tab[(1, 0)],
        "model_WRONG_and_shortest_correct": tab[(0, 1)],
        "model_wrong_and_shortest_wrong": tab[(0, 0)],
    },
    "interpretation": "Among the 63 cases where the shortest option is NOT the answer, the model is right on "
                      f"{tab[(1, 0)]} and wrong on {tab[(0, 0)]}; among the 648 where it IS, the model is wrong on {tab[(0, 1)]}.",
    "model_wrong_when_shortest_right": model_wrong_when_shortest_right,
    "mcnemar_detectability_corrected": detect,
}
json.dump(out, open(f"{AN}/length_artifact.json", "w"), indent=2)
print(json.dumps(out, indent=2))
