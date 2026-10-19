#!/usr/bin/env python
"""A29 analysis: implements exactly the rules L0-L3 and the reading rule of docs/preregistration_v1_amendment_A29.md.
python a29_analysis.py -> analysis/a29_summary.json  (needs scripts/a29_summarize.py to have been run on every run directory)"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SERIES = {"S1": ("q3b", 20261091), "S2": ("q3b", 20261092)}
DOSES = (0.25, 0.50, 0.70, 0.85)
out = {"series": {}}
for name, (fam, seed) in SERIES.items():
    runs = {}
    for d in DOSES:
        f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a29_{fam}_length{d:.2f}_seed{seed}_2*/summary_dose.json") if "smoke" not in x]
        if f:
            runs[d] = json.load(open(f[0]))
    if len(runs) < len(DOSES):
        out["series"][name] = {"family": fam, "seed": seed, "incomplete": sorted(runs)}
        continue
    g = {d: runs[d]["g"]["base"]["g"] for d in DOSES}
    V = {d: runs[d]["V_star_blur"] for d in DOSES}
    pick = {d: runs[d].get("pred_equals_shortest") for d in DOSES}
    L0 = bool(g[0.25] >= 3.0)
    L1 = bool(g[0.25] - g[0.85] >= 3.0 and all(g[a] >= g[b] - 0.5 for a, b in ((0.25, 0.50), (0.50, 0.70), (0.70, 0.85))))
    L2 = bool(pick[0.85] >= 0.85)
    out["series"][name] = {"family": fam, "seed": seed, "g": g, "V_star": V, "pick_shortest": pick, "acc_planted": {d: runs[d]["acc"]["base"] for d in DOSES},
                           "acc_black": {d: runs[d]["acc"]["black"] for d in DOSES}, "orig_E1_acc": {d: runs[d]["orig_E1"]["acc"] for d in DOSES},
                           "L0": L0, "L1": L1, "L2": L2, "L3_main_g085_le2": bool(g[0.85] <= 2.0), "L3_strong_g070_le2": bool(g[0.70] <= 2.0)}
done = {k: v for k, v in out["series"].items() if "incomplete" not in v}
if len(done) == len(SERIES):
    both = lambda key: all(v[key] for v in done.values())
    out["L0_L1_L2_both"] = bool(both("L0") and both("L1") and both("L2"))
    out["L3_main_both"] = both("L3_main_g085_le2")
    out["L3_strong_both"] = both("L3_strong_g070_le2")
    first = {}
    for k, v in done.items():
        first[k] = next((d for d in DOSES if v["g"][d] <= 2.0), None)
    out["first_dose_with_g_le_2"] = first
    if not out["L0_L1_L2_both"]:
        out["verdict"] = "not informative: the dose-response itself (L0-L2) did not hold in both series; no lock-out claim for OmniMedVQA"
    elif out["L3_strong_both"]:
        out["verdict"] = "supports: lock-out on OmniMedVQA by d=0.70, earlier than PMC-VQA (g=+3.4 at 0.70), in the order of the benefit without a cue"
    elif out["L3_main_both"]:
        out["verdict"] = "partly supports: lock-out by d=0.85 as on PMC-VQA, not earlier; the benefit order OmniMedVQA < PMC-VQA is not resolved"
    else:
        out["verdict"] = "does not support: the benefit without a cue does not predict the lock-out strength; reported as a failure of the predictor"
json.dump(out, open(ROOT + "/analysis/a29_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
