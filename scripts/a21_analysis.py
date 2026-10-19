#!/usr/bin/env python
"""A21 analysis: implements exactly the rules R1-R4 and the reading rule of docs/preregistration_v1_amendment_A21.md.
python a21_analysis.py -> analysis/a21_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SERIES = {"S1": ("q3b", 20261021), "S2": ("q3b", 20261022), "S3": ("ivl2b", 20261021)}
DOSES = (0.25, 0.70, 0.92)
out = {"series": {}}
for name, (fam, seed) in SERIES.items():
    runs = {}
    for d in DOSES:
        f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a21_{fam}_length{d:.2f}_seed{seed}_2*/summary_dose.json") if "smoke" not in x]
        if f:
            runs[d] = json.load(open(f[0]))
    if len(runs) < 3:
        out["series"][name] = {"family": fam, "seed": seed, "incomplete": sorted(runs)}
        continue
    g = {d: runs[d]["g"]["base"]["g"] for d in DOSES}
    V = {d: runs[d]["V_star_blur"] for d in DOSES}
    pick = {d: runs[d].get("pred_equals_shortest") for d in DOSES}
    orig = {d: runs[d]["orig_E1"]["acc"] for d in DOSES}
    R1 = bool(g[0.25] - g[0.92] >= 5.0 and g[0.25] > g[0.70] > g[0.92])
    R2 = bool(pick[0.92] >= 0.85 and pick[0.25] <= 0.35 and orig[0.25] - orig[0.92] >= 5.0)
    R3 = bool(V[0.92] >= V[0.25] + 1)
    R4 = bool(g[0.92] <= 2.0 and V[0.92] == 7)
    out["series"][name] = {"family": fam, "seed": seed, "g": g, "V_star": V, "pick_shortest": pick, "orig_E1_acc": orig, "acc_planted": {d: runs[d]["acc"]["base"] for d in DOSES},
                           "acc_black": {d: runs[d]["acc"]["black"] for d in DOSES}, "median_Mt": {d: runs[d]["median_logodds_margin_black"] for d in DOSES},
                           "ladder_g": {d: {a: runs[d]["g"][a]["g"] for a in ("blur32", "blur16", "blur8", "blur4", "blur2", "base")} for d in DOSES},
                           "R1": R1, "R2": R2, "R3": R3, "R4_strong": R4, "R1_R2_R3": bool(R1 and R2 and R3)}
done = {k: v for k, v in out["series"].items() if "incomplete" not in v}
if len(done) == len(SERIES):
    qwen = all(done[k]["R1_R2_R3"] for k in ("S1", "S2"))
    ivl = done["S3"]["R1_R2_R3"]
    out["verdict"] = ("replicates (all three series)" if qwen and ivl else "partly replicates (both Qwen series, not InternVL3-2B)" if qwen else
                      "partly replicates (InternVL3-2B, not both Qwen series)" if ivl else "does not replicate" if not done["S1"]["R1"] else "mixed: see per-series rules")
    out["R4_strong_by_series"] = {k: v["R4_strong"] for k, v in done.items()}
json.dump(out, open(ROOT + "/analysis/a21_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
