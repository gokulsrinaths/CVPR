#!/usr/bin/env python
"""
Phase 2b (CPU only): (1) base rates of the error-flag heuristics over ALL 711 MCQs with Fisher exact tests,
(2) trivial text-only baselines (most-central option / longest / shortest) to detect dataset artifacts,
(3) exact-McNemar detectability table (how many corrections are needed for significance at 96% baseline).
Reads analysis/mcq_all711_features.csv only. No model, no GPU.
"""
import collections
import csv
import json
import math
import os
import re
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
AN = f"{ROOT}/analysis"
sys.path.insert(0, f"{ROOT}/scoring_suite/scoring")
from medreason_scoring.normalize import tokens

rows = list(csv.DictReader(open(f"{AN}/mcq_all711_features.csv")))
assert len(rows) == 711
err = [r for r in rows if r["correct"] == "0"]
ok = [r for r in rows if r["correct"] == "1"]


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] exact two-sided p (sum of tables with prob <= observed)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    obs = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12)


# ---------- (1) flag base rates
def flagset(r):
    return set(filter(None, r["suspected_issue"].split(";")))


flag_names = ["near_duplicate_options(gt~pred)", "highly_overlapping_option_set", "ood_routed_preprocessing",
              "near_detector_boundary", "pred_A_could_be_parser_default", "multi_modality_label",
              "rare_modality(<2%)", "no_visual_cue_in_question(knowledge_like)"]


def has(r, f):
    # recompute flags over ALL rows (csv suspected_issue for correct rows is also populated by the audit script)
    return f in flagset(r)


base = {}
for f in flag_names:
    a = sum(has(r, f) for r in err)
    b = len(err) - a
    c = sum(has(r, f) for r in ok)
    d = len(ok) - c
    base[f] = {"errors_with_flag": f"{a}/{len(err)}", "correct_with_flag": f"{c}/{len(ok)}",
               "rate_errors": round(a / len(err), 3), "rate_correct": round(c / len(ok), 3),
               "fisher_p": round(fisher_two_sided(a, b, c, d), 5)}
# jaccard(gt,pred) is undefined for correct rows (gt==pred => 1.0); use max_pairwise_option_jaccard for enrichment
hi_pair = lambda r: float(r["max_pairwise_option_jaccard"]) >= 0.6
a = sum(hi_pair(r) for r in err); c = sum(hi_pair(r) for r in ok)
base["highly_overlapping_option_set (recomputed)"] = {
    "errors_with_flag": f"{a}/{len(err)}", "correct_with_flag": f"{c}/{len(ok)}",
    "rate_errors": round(a / len(err), 3), "rate_correct": round(c / len(ok), 3),
    "fisher_p": round(fisher_two_sided(a, len(err) - a, c, len(ok) - c), 5)}
vis = lambda r: r["question_visual_cue"] == "1"
a = sum(not vis(r) for r in err); c = sum(not vis(r) for r in ok)
base["no_visual_cue_in_question (recomputed)"] = {
    "errors_with_flag": f"{a}/{len(err)}", "correct_with_flag": f"{c}/{len(ok)}",
    "rate_errors": round(a / len(err), 3), "rate_correct": round(c / len(ok), 3),
    "fisher_p": round(fisher_two_sided(a, len(err) - a, c, len(ok) - c), 5)}

# ---------- (2) trivial text-only baselines
def parse_opts(s):
    out = {}
    for part in s.split(" || "):
        m = re.match(r"^([A-E]): (.*)$", part, re.S)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def jacc(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / len(a | b) if (a | b) else 0.0


hits = collections.Counter()
model_agree = collections.Counter()
model_agree_err = collections.Counter()
for r in rows:
    o = parse_opts(r["options"])
    g, p = r["ground_truth"], r["parsed_prediction"]
    cent = {k: sum(jacc(o[k], o[j]) for j in o if j != k) / (len(o) - 1) for k in o}
    picks = {
        "most_central_option": max(cent, key=lambda k: (cent[k], -ord(k))),
        "longest_option": max(o, key=lambda k: (len(o[k]), -ord(k))),
        "shortest_option": min(o, key=lambda k: (len(o[k]), ord(k))),
        "always_A": "A",
    }
    for name, pick in picks.items():
        hits[name] += (pick == g)
        model_agree[name] += (pick == p)
        if r["correct"] == "0":
            model_agree_err[name] += (pick == p)
text_baselines = {
    name: {"accuracy": round(hits[name] / 711, 4), "correct": hits[name],
           "model_picks_same_option_overall": round(model_agree[name] / 711, 4),
           "model_picks_same_option_on_its_28_errors": f"{model_agree_err[name]}/28"}
    for name in hits}
text_baselines["chance_5_options"] = {"accuracy": 0.2}

# ---------- (3) exact McNemar detectability at a 683/711 baseline
def mcnemar_p(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


detect = {}
for b in (0, 1, 2, 3, 5, 8):
    need = next((f for f in range(0, 29) if mcnemar_p(f, b) < 0.05), None)
    detect[f"baseline_correct_cases_broken={b}"] = {
        "min_baseline_errors_fixed_for_p<0.05": need,
        "net_accuracy_gain_pp": None if need is None else round(100 * (need - b) / 711, 2),
        "max_possible_fixes": 28}
out = {"n_errors": len(err), "n_correct": len(ok), "flag_base_rates_with_fisher": base,
       "text_only_baselines_over_711": text_baselines,
       "mcnemar_detectability_at_683_of_711": detect,
       "ceiling_note": "With 28 errors the maximum possible net gain is +3.94pp (28/711); a method that breaks "
                       "only a handful of currently-correct cases loses most of the significance budget."}
json.dump(out, open(f"{AN}/baseline_artifact_checks.json", "w"), indent=2)
print(json.dumps(out, indent=2))
