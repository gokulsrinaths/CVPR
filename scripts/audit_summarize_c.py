#!/usr/bin/env python
"""Summary for H2 (docs/gate_C_prespec.md). python audit_summarize_c.py RUN_DIR"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, mcnemar_exact, gold_from_train, shortest_letter, argmax, LETTERS

run_dir = sys.argv[1]
rows = read_jsonl(os.path.join(run_dir, "forward_rows.jsonl"))
T = gold_from_train()
arm = {"base": {}, "black": {}}
for r in rows:
    arm[r["arm"]][r["case_id"]] = r
ids = sorted(set(arm["base"]) & set(arm["black"]))
n = len(ids)
short = {i: shortest_letter(T[i]) for i in ids}
gold = {i: T[i]["answer"].strip() for i in ids}
ws = [i for i in ids if short[i] != gold[i]]
out = {"run_dir": run_dir, "n_cases": n, "label": "DEV split is IN-SAMPLE SUSPECTED (Gate B J1 accuracy trigger); not used for calibration", "n_shortest_wrong": len(ws)}
for ro in ("direct", "cue"):
    pred = lambda a, i: LETTERS[argmax(arm[a][i][ro]["lp"])]
    tb = sum(pred("base", i) == gold[i] for i in ids)
    tk = sum(pred("black", i) == gold[i] for i in ids)
    b = sum(pred("base", i) == gold[i] and pred("black", i) != gold[i] for i in ids)
    c = sum(pred("base", i) != gold[i] and pred("black", i) == gold[i] for i in ids)
    out[ro] = {"acc_true": round(tb / n, 4), "wilson_true": wilson(tb, n), "acc_black": round(tk / n, 4), "wilson_black": wilson(tk, n),
               "loss_points": round(100 * (tb - tk) / n, 2), "mcnemar_true_only": b, "mcnemar_black_only": c, "mcnemar_p": mcnemar_exact(b, c),
               "share_pred_equals_shortest_true": round(sum(pred("base", i) == short[i] for i in ids) / n, 4),
               "share_pred_equals_shortest_black": round(sum(pred("black", i) == short[i] for i in ids) / n, 4),
               "acc_true_on_shortest_wrong": round(sum(pred("base", i) == gold[i] for i in ws) / max(1, len(ws)), 4),
               "acc_black_on_shortest_wrong": round(sum(pred("black", i) == gold[i] for i in ws) / max(1, len(ws)), 4)}
out["H2"] = "SUPPORTED" if out["direct"]["loss_points"] <= 5.0 else "FALSIFIED"
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
