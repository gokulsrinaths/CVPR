#!/usr/bin/env python
"""A26 analysis: exactly rules F1, F2 and the reading rule of docs/preregistration_v1_amendment_A26.md. python a26_analysis.py -> analysis/a26_summary.json
Uses scripts/a12_summarize.py m2 output (summary_a12.json) of each run."""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SPEC = {"ul": ("ul", (20261041, 20261042, 20261043)), "lpul": ("lpul", (20261041, 20261042, 20261043)), "both": ("both", (20261031, 20261032, 20261033, 20261034, 20261035))}
out = {"methods": {}}
for m, (pre, seeds) in SPEC.items():
    runs = []
    for sd in seeds:
        pat = f"{ROOT}/experiments/pilot/a2[46]_medcmr_{pre}_3b_seed{sd}_2*/summary_a12.json"
        f = [x for x in glob.glob(pat) if "smoke" not in x]
        if not f:
            continue
        s = json.load(open(sorted(f)[-1]))
        al = s["FRESH"]["all"]
        acc = 100 * s["FRESH"]["acc_true"]
        f1 = bool(al["diff_points"] >= 3.0 and al["mcnemar_p"] < 0.025)
        f2 = bool(acc >= 88.0)
        runs.append({"seed": sd, "gap_all": al["diff_points"], "p": al["mcnemar_p"], "fixes": al["fixes"], "breaks": al["breaks"], "fresh_acc": acc, "test_acc": 100 * s["TEST711"]["acc_true"], "shortest_pick": s["TEST711"]["pred_equals_shortest"],
                     "HI_changed": s["own_strata"]["HI_changed"], "F1": f1, "F2": f2, "F1_and_F2": bool(f1 and f2)})
    n_ok = sum(r["F1_and_F2"] for r in runs)
    out["methods"][m] = {"runs": runs, "n_runs": len(runs), "n_seeds_F1_and_F2": n_ok}
    if m in ("ul", "lpul"):
        out["methods"][m]["success"] = bool(len(runs) == 3 and n_ok >= 2)
json.dump(out, open(ROOT + "/analysis/a26_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
