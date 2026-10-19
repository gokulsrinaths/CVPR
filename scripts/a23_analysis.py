#!/usr/bin/env python
"""A23 analysis: exactly rules B1, B2, C1-C3 of docs/preregistration_v1_amendment_A23.md. python a23_analysis.py -> analysis/a23_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"


def load(fam, d, seed):
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a21_{fam}_length{d:.2f}_seed{seed}_2*/summary_dose.json") if "smoke" not in x]
    return json.load(open(sorted(f)[-1])) if f else None


def row(r):
    return {"g": r["g"]["base"]["g"], "ci95": r["g"]["base"]["ci95"], "V": r["V_star_blur"], "pick": r.get("pred_equals_shortest"), "orig_acc": r["orig_E1"]["acc"], "acc": r["acc"]["base"], "acc_black": r["acc"]["black"],
            "ladder": {a: r["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")}}


out = {}
b = {d: load("q3b", d, 20261022) for d in (0.92, 0.97, 0.99)}
if all(b.values()):
    R = {d: row(b[d]) for d in b}
    out["B"] = {"runs": R, "B1": bool(R[0.99]["g"] <= 2.0 and R[0.99]["V"] == 7), "B2": bool(R[0.92]["g"] > R[0.97]["g"] > R[0.99]["g"])}
else:
    out["B"] = {"incomplete": [d for d in b if b[d] is None]}
c = {d: load("q7b", d, 20261021) for d in (0.25, 0.70, 0.92, 0.99)}
if all(c.values()):
    R = {d: row(c[d]) for d in c}
    out["C"] = {"runs": R, "C1": bool(R[0.25]["g"] - R[0.92]["g"] >= 5.0 and R[0.25]["g"] > R[0.70]["g"] > R[0.92]["g"]), "C2": bool(R[0.92]["V"] >= R[0.25]["V"] + 1),
                "C3_lockout_reported": bool(R[0.99]["g"] <= 2.0 and R[0.99]["V"] == 7)}
else:
    out["C"] = {"incomplete": [d for d in c if c[d] is None]}
json.dump(out, open(ROOT + "/analysis/a23_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
