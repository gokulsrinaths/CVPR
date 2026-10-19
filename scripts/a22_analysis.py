#!/usr/bin/env python
"""A22 analysis: implements exactly the rules H1-H3, P1-P4 and S (rules A, B, C of A19) of docs/preregistration_v1_amendment_A22.md.
python a22_analysis.py -> analysis/a22_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
G_PRED = {0.25: 13.5, 0.70: 3.4, 0.92: 0.1}


def load(pattern):
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pattern}/summary_dose.json") if "smoke" not in x]
    return json.load(open(sorted(f)[-1])) if f else None


out = {}
# Part H: higher doses of the length shortcut, Qwen-3B seed 20261021 (A21 runs at 0.92 and the new runs at 0.97 and 0.99)
h = {d: load(f"a21_q3b_length{d:.2f}_seed20261021_2*") for d in (0.25, 0.70, 0.92, 0.97, 0.99)}
if all(h[d] is not None for d in (0.92, 0.97, 0.99)):
    g = {d: h[d]["g"]["base"]["g"] for d in h if h[d]}
    V = {d: h[d]["V_star_blur"] for d in h if h[d]}
    pick = {d: h[d].get("pred_equals_shortest") for d in h if h[d]}
    out["H"] = {"g": g, "V_star": V, "pick_shortest": pick, "H1": bool(g[0.99] <= g[0.92] - 3.0), "H2": bool(pick[0.99] >= 0.95), "H3_lockout_reported": bool(g[0.99] <= 2.0 and V[0.99] == 7),
                "ladder_g": {d: {a: h[d]["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")} for d in h if h[d]}}
else:
    out["H"] = {"incomplete": [d for d in h if h[d] is None]}
# Part P: position shortcut, Qwen-3B seed 20261021
p = {d: load(f"a21_q3b_position{d:.2f}_seed20261021_2*") for d in (0.25, 0.70, 0.92)}
if all(p.values()):
    g = {d: p[d]["g"]["base"]["g"] for d in p}
    V = {d: p[d]["V_star_blur"] for d in p}
    pa = {d: p[d].get("pred_equals_A") for d in p}
    out["P"] = {"g": g, "V_star": V, "pick_A": pa, "P1": bool(g[0.25] - g[0.92] >= 5.0 and g[0.25] > g[0.70] > g[0.92]), "P2": bool(pa[0.92] >= 0.85 and pa[0.25] <= 0.35), "P3": bool(V[0.92] >= V[0.25] + 1),
                "P4_lockout_reported": bool(g[0.92] <= 2.0 and V[0.92] == 7), "ladder_g": {d: {a: p[d]["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")} for d in p}}
else:
    out["P"] = {"incomplete": [d for d in p if p[d] is None]}
# Part S: second seed of A19 (rules A, B, C unchanged)
out["S"] = {}
for fam in ("ivl2b", "q7b"):
    runs = {d: load(f"dose_{fam}_length{d:.2f}_seed20261012_2*") for d in (0.25, 0.70, 0.92)}
    if not all(runs.values()):
        out["S"][fam] = {"incomplete": [d for d in runs if runs[d] is None]}
        continue
    g = {d: runs[d]["g"]["base"]["g"] for d in runs}
    V = {d: runs[d]["V_star_blur"] for d in runs}
    A = all(abs(g[d] - G_PRED[d]) <= 4.0 for d in G_PRED)
    B = bool(V[0.25] <= 6 and V[0.92] == 7)
    C = bool(g[0.25] - g[0.70] >= 2.0 and g[0.70] - g[0.92] >= 1.0 and g[0.92] <= 2.0)
    out["S"][fam] = {"g_obs": g, "abs_dev": {d: round(abs(g[d] - G_PRED[d]), 2) for d in G_PRED}, "V_star": V, "A": bool(A), "B": B, "C": C, "passes": bool(A and B and C),
                     "pick_shortest": {d: runs[d].get("pred_equals_shortest") for d in runs}, "orig_acc": {d: runs[d]["orig_E1"]["acc"] for d in runs}}
json.dump(out, open(ROOT + "/analysis/a22_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
