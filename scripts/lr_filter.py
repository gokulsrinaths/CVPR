#!/usr/bin/env python
"""A6 step 2 (CPU): gate G1 on TEST711, eq_options.json for the passing questions, and the 50-question hand-check sample for G2.
python lr_filter.py RUN_DIR"""
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_common as C
from audit_summarize import gold_from_train

run_dir = sys.argv[1]
R = json.load(open(os.path.join(run_dir, "rewrites.json")))
T = gold_from_train()
out = {"run_dir": run_dir}
eq = {"test711": {}, "dev700": {}}
for k, v in R.items():
    g, cid = k.split("|")
    if v["passed"]:
        eq[g][cid] = v["new"]
json.dump(eq, open(os.path.join(run_dir, "eq_options.json"), "w"))


def shortest_char(opts):
    return min(C.LETTERS, key=lambda l: (len(opts[l]), l))


for g in ("test711", "dev700"):
    ks = [k for k in R if k.startswith(g + "|")]
    ps = [k for k in ks if R[k]["passed"]]
    ratios = [max(C.words(R[k]["new"][l]) for l in C.LETTERS) / max(1, min(C.words(R[k]["new"][l]) for l in C.LETTERS)) for k in ps]
    o_ratios = [max(C.words(R[k]["orig"][l]) for l in C.LETTERS) / max(1, min(C.words(R[k]["orig"][l]) for l in C.LETTERS)) for k in ps]
    gold = {k: T[k.split("|")[1]]["answer"].strip() for k in ps}
    rule_after = sum(shortest_char(R[k]["new"]) == gold[k] for k in ps) / max(1, len(ps))
    rule_before = sum(shortest_char(R[k]["orig"]) == gold[k] for k in ps) / max(1, len(ps))
    gold_h_b = st.mean(C.hedges(R[k]["orig"][gold[k]]) for k in ps)
    dis_h_b = st.mean(st.mean(C.hedges(R[k]["orig"][l]) for l in C.LETTERS if l != gold[k]) for k in ps)
    gold_h_a = st.mean(C.hedges(R[k]["new"][gold[k]]) for k in ps)
    dis_h_a = st.mean(st.mean(C.hedges(R[k]["new"][l]) for l in C.LETTERS if l != gold[k]) for k in ps)
    why = {}
    for k in ks:
        if not R[k]["passed"]:
            for r in R[k]["reasons"]:
                r = r.split("_")[0]
                why[r] = why.get(r, 0) + 1
    out[g] = {"n": len(ks), "passed": len(ps), "pass_rate": round(len(ps) / len(ks), 4), "attempt2_used": sum(R[k]["attempt"] == 2 for k in ks),
              "fail_reasons_final": why, "median_max_over_min_words_after": round(st.median(ratios), 3), "median_max_over_min_words_before": round(st.median(o_ratios), 3),
              "shortest_char_rule_acc_before": round(rule_before, 4), "shortest_char_rule_acc_after": round(rule_after, 4),
              "hedges_per_option_gold_before_after": [round(gold_h_b, 3), round(gold_h_a, 3)], "hedges_per_option_distractor_before_after": [round(dis_h_b, 3), round(dis_h_a, 3)]}
t = out["test711"]
out["G1"] = {"pass_rate_ge_0.60": t["pass_rate"] >= 0.60, "median_ratio_le_1.20": t["median_max_over_min_words_after"] <= 1.20, "shortest_rule_acc_le_0.40": t["shortest_char_rule_acc_after"] <= 0.40}
out["G1_pass"] = all(out["G1"].values())
passing_test = sorted(k.split("|")[1] for k in R if k.startswith("test711|") and R[k]["passed"])
sample = random.Random(20261003).sample(passing_test, min(50, len(passing_test)))
hc = [{"case_id": c, "gold": T[c]["answer"].strip(), "question": T[c]["question"], "orig": R["test711|" + c]["orig"], "new": R["test711|" + c]["new"]} for c in sample]
json.dump(hc, open(os.path.join(run_dir, "handcheck_sample.json"), "w"), indent=1)
json.dump(out, open(os.path.join(run_dir, "filter_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
