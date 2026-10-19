#!/usr/bin/env python
"""A19 analysis: held-out transfer of the Qwen-3B dose-response. Implements exactly the rules A, B, C of docs/preregistration_v1_amendment_A19.md.
python a19_analysis.py -> analysis/a19_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
G_PRED = {0.25: 13.5, 0.70: 3.4, 0.92: 0.1}
out = {"g_pred": G_PRED, "models": {}}
for fam in ("ivl2b", "q7b"):
    runs = {}
    for d in (0.25, 0.70, 0.92):
        f = glob.glob(f"{ROOT}/experiments/pilot/dose_{fam}_length{d:.2f}_seed20261011_2*/summary_dose.json")
        f = [x for x in f if "smoke" not in x]
        if f:
            runs[d] = json.load(open(f[0]))
    if len(runs) < 3:
        out["models"][fam] = {"incomplete": sorted(runs)}
        continue
    g = {d: runs[d]["g"]["base"]["g"] for d in runs}
    V = {d: runs[d]["V_star_blur"] for d in runs}
    A = all(abs(g[d] - G_PRED[d]) <= 4.0 for d in G_PRED)
    B = bool(V[0.25] <= 6 and V[0.92] == 7)
    C = bool(g[0.25] - g[0.70] >= 2.0 and g[0.70] - g[0.92] >= 1.0 and g[0.92] <= 2.0)
    out["models"][fam] = {"g_obs": g, "abs_dev": {d: round(abs(g[d] - G_PRED[d]), 2) for d in G_PRED}, "V_star": V, "A_quantitative": bool(A), "B_boundary": B, "C_monotone": C, "passes": bool(A and B and C),
                          "median_Mt": {d: runs[d]["median_logodds_margin_black"] for d in runs}, "pick_shortest": {d: runs[d].get("pred_equals_shortest") for d in runs},
                          "acc": {d: runs[d]["acc"]["base"] for d in runs}, "orig_acc": {d: runs[d]["orig_E1"]["acc"] for d in runs},
                          "ladder_g": {d: {a: runs[d]["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")} for d in runs}}
json.dump(out, open(ROOT + "/analysis/a19_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
