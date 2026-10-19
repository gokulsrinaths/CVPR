#!/usr/bin/env python
"""A6 analysis. python lr_summarize.py EVAL_RUN_DIR REWRITE_RUN_DIR"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, mcnemar_exact, gold_from_train, shortest_letter, argmax, LETTERS

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
ev, rw = sys.argv[1], sys.argv[2]
T = gold_from_train()
rows = read_jsonl(os.path.join(ev, "rows.jsonl"))
R = json.load(open(os.path.join(rw, "rewrites.json")))
orig_lp = {"test711": {}, "dev700": {}}
for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
    for r in read_jsonl(pth):
        if r["arm"] == "base":
            orig_lp["test711"][r["case_id"]] = r
for r in read_jsonl(glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/forward_rows.jsonl")[0]):
    if r["arm"] == "base":
        orig_lp["dev700"][r["case_id"]] = r
arm = {"test711": {"eq": {}, "eq_black": {}}, "dev700": {"eq": {}, "eq_black": {}}}
for r in rows:
    arm[r["group"]][r["arm"]][r["case_id"]] = r
out = {"eval_run": ev, "rewrite_run": rw}
for g in ("test711", "dev700"):
    ids = sorted(set(arm[g]["eq"]) & set(arm[g]["eq_black"]))
    n = len(ids)
    gold = {i: T[i]["answer"].strip() for i in ids}
    short_o = {i: shortest_letter(T[i]) for i in ids}
    ws = [i for i in ids if short_o[i] != gold[i]]
    res = {"n_passing_cases": n, "n_shortest_wrong_originally": len(ws)}
    for ro in ("direct", "cue"):
        po = {i: LETTERS[argmax(orig_lp[g][i][ro]["lp"])] for i in ids}
        pe = {i: LETTERS[argmax(arm[g]["eq"][i][ro]["lp"])] for i in ids}
        pb = {i: LETTERS[argmax(arm[g]["eq_black"][i][ro]["lp"])] for i in ids}
        ko = sum(po[i] == gold[i] for i in ids)
        ke = sum(pe[i] == gold[i] for i in ids)
        kb = sum(pb[i] == gold[i] for i in ids)
        f = sum(pe[i] == gold[i] and po[i] != gold[i] for i in ids)
        b = sum(pe[i] != gold[i] and po[i] == gold[i] for i in ids)
        f2 = sum(pe[i] == gold[i] and pb[i] != gold[i] for i in ids)
        b2 = sum(pe[i] != gold[i] and pb[i] == gold[i] for i in ids)
        res[ro] = {"acc_original": round(ko / n, 4), "acc_eq": round(ke / n, 4), "wilson_eq": wilson(ke, n), "acc_eq_black": round(kb / n, 4),
                   "drop_points": round(100 * (ko - ke) / n, 2), "original_right_eq_wrong": b, "original_wrong_eq_right": f, "mcnemar_p_orig_vs_eq": mcnemar_exact(f, b),
                   "image_effect_on_eq_points": round(100 * (ke - kb) / n, 2), "eq_true_only": f2, "eq_black_only": b2, "mcnemar_p_eq_true_vs_black": mcnemar_exact(f2, b2),
                   "share_pred_equals_original_shortest_orig": round(sum(po[i] == short_o[i] for i in ids) / n, 4),
                   "share_pred_equals_original_shortest_eq": round(sum(pe[i] == short_o[i] for i in ids) / n, 4),
                   "acc_original_on_orig_shortest_wrong": round(sum(po[i] == gold[i] for i in ws) / max(1, len(ws)), 4),
                   "acc_eq_on_orig_shortest_wrong": round(sum(pe[i] == gold[i] for i in ws) / max(1, len(ws)), 4)}
    out[g] = res


def l_rule(d):
    drop, p = d["direct"]["drop_points"], d["direct"]["mcnemar_p_orig_vs_eq"]
    if drop >= 5.0 and p < 0.05:
        return "RELIANCE SUPPORTED"
    if drop <= 2.0:
        return "NO RELIANCE (shortcut story falsified for this model)"
    return "INCONCLUSIVE"


out["L1_test"] = l_rule(out["test711"])
out["L2_dev"] = l_rule(out["dev700"])
t = out["test711"]["direct"]
out["L3_image_matters_more"] = "NOT EVALUABLE (acc_eq < 0.40)" if t["acc_eq"] < 0.40 else ("SUPPORTED" if t["image_effect_on_eq_points"] >= 5.0 else "FALSIFIED")
json.dump(out, open(os.path.join(ev, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
