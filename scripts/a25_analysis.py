#!/usr/bin/env python
"""A25 analysis: exactly rules D1, D2, L1, L2 of docs/preregistration_v1_amendment_A25.md. python a25_analysis.py -> analysis/a25_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"


def load(fam, d, seed):
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a21_{fam}_length{d:.2f}_seed{seed}_2*/summary_dose.json") if "smoke" not in x]
    return json.load(open(sorted(f)[-1])) if f else None


def row(r):
    return {"g": r["g"]["base"]["g"], "ci95": r["g"]["base"]["ci95"], "V": r["V_star_blur"], "pick": r.get("pred_equals_shortest"), "orig_acc": r["orig_E1"]["acc"], "acc": r["acc"]["base"], "acc_black": r["acc"]["black"],
            "ladder": {a: r["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")}}


out = {"series": {}}
for name, fam, seed in (("S1 q7b seed 20261022", "q7b", 20261022), ("S2 ivl2b seed 20261022", "ivl2b", 20261022)):
    c = {d: load(fam, d, seed) for d in (0.25, 0.70, 0.92, 0.99)}
    if not all(c.values()):
        out["series"][name] = {"incomplete": [d for d in c if c[d] is None]}
        continue
    R = {d: row(c[d]) for d in c}
    out["series"][name] = {"runs": R, "D1": bool(R[0.25]["g"] - R[0.92]["g"] >= 5.0 and R[0.25]["g"] > R[0.70]["g"] > R[0.92]["g"]), "D2": bool(R[0.92]["V"] >= R[0.25]["V"] + 1),
                           "L1": bool(R[0.99]["g"] <= 2.0 and R[0.99]["V"] == 7)}
b = {d: load("q3b", d, 20261023) for d in (0.92, 0.97, 0.99)}
if all(b.values()):
    R = {d: row(b[d]) for d in b}
    out["series"]["S3 q3b seed 20261023"] = {"runs": R, "L1": bool(R[0.99]["g"] <= 2.0 and R[0.99]["V"] == 7), "L2": bool(R[0.92]["g"] > R[0.97]["g"] > R[0.99]["g"])}
else:
    out["series"]["S3 q3b seed 20261023"] = {"incomplete": [d for d in b if b[d] is None]}
json.dump(out, open(ROOT + "/analysis/a25_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
