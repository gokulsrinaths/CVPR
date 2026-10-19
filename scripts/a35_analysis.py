#!/usr/bin/env python
"""A35 analysis: exactly rules O1, O2 and the reading rule of docs/preregistration_v1_amendment_A35.md. python a35_analysis.py -> analysis/a35_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
RUNS = {"pmc_0.25_s1": "pmc025s11", "pmc_0.25_s2": "pmc025s12", "pmc_0.25_s3": "pmc025s13", "pmc_0.70_s1": "pmc070s11", "pmc_0.70_s2": "pmc070s12", "pmc_0.70_s3": "pmc070s13",
        "aokvqa_0.25_s1": "aok025s21", "aokvqa_0.25_s2": "aok025s22"}
out = {"runs": {}}
for name, lab in RUNS.items():
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a35_occl_{lab}_2*/rows_occl.jsonl") if "smoke" not in x]
    if not f:
        out["runs"][name] = {"incomplete": True}
        continue
    P = {}
    for ln in open(f[0]):
        r = json.loads(ln)
        P.setdefault(r["arm"], {})[r["case_id"]] = int(max(range(4), key=lambda k: r["lp"][k]) == "ABCD".index(r["gold"]))
    ids = sorted(set.intersection(*[set(P[a]) for a in P]))
    acc = {a: round(100 * sum(P[a][i] for i in ids) / len(ids), 2) for a in P}
    g = acc["base"] - acc["black"]
    out["runs"][name] = {"n": len(ids), "acc": acc, "benefit_true_minus_black": round(g, 2), "center_minus_border_acc": round(acc["occl_center"] - acc["occl_border"], 2),
                         "share_of_benefit_lost_center": round((acc["base"] - acc["occl_center"]) / g, 3) if abs(g) > 1 else None,
                         "share_of_benefit_lost_border": round((acc["base"] - acc["occl_border"]) / g, 3) if abs(g) > 1 else None}
R = out["runs"]
if all("incomplete" not in R[k] for k in ("pmc_0.25_s1", "pmc_0.25_s2", "pmc_0.25_s3", "aokvqa_0.25_s1", "aokvqa_0.25_s2")):
    out["O1_pmc_central_loses_ge_2_more_in_2_of_3"] = bool(sum(R[f"pmc_0.25_s{i}"]["center_minus_border_acc"] <= -2.0 for i in (1, 2, 3)) >= 2)
    out["O2_aokvqa_central_loses_ge_2_more_in_both"] = bool(all(R[f"aokvqa_0.25_s{i}"]["center_minus_border_acc"] <= -2.0 for i in (1, 2)))
json.dump(out, open(ROOT + "/analysis/a35_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
