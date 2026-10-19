#!/usr/bin/env python
"""Summary for A4 (Qwen2.5-VL-7B audit). python audit_summarize_s2.py RUN_DIR"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, mcnemar_exact, gold_from_train, shortest_letter

run_dir = sys.argv[1]
rows = read_jsonl(os.path.join(run_dir, "rows.jsonl"))
T = gold_from_train()
out = {"run_dir": run_dir}
for g in ("test711", "dev700"):
    arm = {"base": {}, "black": {}}
    for r in rows:
        if r["group"] == g:
            arm[r["arm"]][r["case_id"]] = r
    ids = sorted(set(arm["base"]) & set(arm["black"]))
    n = len(ids)
    if n == 0:
        out[g] = {"n": 0}
        continue
    short = {i: shortest_letter(T[i]) for i in ids}
    gold = {i: T[i]["answer"].strip() for i in ids}
    ws = [i for i in ids if short[i] != gold[i]]
    pr = lambda a, i: arm[a][i]["parsed"]
    d = {"n": n, "n_shortest_wrong": len(ws)}
    for a in ("base", "black"):
        k = sum(pr(a, i) == gold[i] for i in ids)
        d[a] = {"correct": k, "acc": round(k / n, 4), "wilson95": wilson(k, n),
                "unparsed_rate": round(sum(pr(a, i) is None for i in ids) / n, 4),
                "share_pred_equals_shortest": round(sum(pr(a, i) == short[i] for i in ids) / n, 4),
                "acc_on_shortest_wrong": round(sum(pr(a, i) == gold[i] for i in ws) / max(1, len(ws)), 4)}
    b = sum(pr("base", i) == gold[i] and pr("black", i) != gold[i] for i in ids)
    c = sum(pr("base", i) != gold[i] and pr("black", i) == gold[i] for i in ids)
    d["loss_points"] = round(100 * (d["base"]["acc"] - d["black"]["acc"]), 2)
    d["mcnemar_true_only"], d["mcnemar_black_only"], d["mcnemar_p"] = b, c, mcnemar_exact(b, c)
    d["raw_examples_true"] = [arm["base"][i]["raw"] for i in ids[:6]]
    out[g] = d
t = out["test711"]
if t.get("n"):
    ev = t["base"]["acc"] >= 0.60 and max(t["base"]["unparsed_rate"], t["black"]["unparsed_rate"]) <= 0.10
    out["S2a_evaluable"] = ev
    out["S2b_test"] = ("SUPPORTED" if t["loss_points"] <= 5.0 else "FALSIFIED") if ev else "NOT EVALUABLE"
    out["S2b_dev"] = ("SUPPORTED" if out["dev700"]["loss_points"] <= 5.0 else "FALSIFIED") if ev and out["dev700"].get("n") else "NOT EVALUABLE"
    out["S2c_shortest_share_ge_0.50"] = ("SUPPORTED" if t["base"]["share_pred_equals_shortest"] >= 0.50 else "FALSIFIED") if ev else "NOT EVALUABLE"
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
