#!/usr/bin/env python
"""A24 analysis: exactly rule E1 of docs/preregistration_v1_amendment_A24.md, per method. python a24_analysis.py -> analysis/a24_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SEEDS = (20261031, 20261032, 20261033)
out = {"methods": {}}
for m in ("lenprior", "mu", "both"):
    runs = []
    for sd in SEEDS:
        f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a24_medcmr_{m}_3b_seed{sd}_2*/summary_a12.json") if "smoke" not in x]
        if not f:
            continue
        s = json.load(open(sorted(f)[-1]))
        lo = s["on_a9_LO_set"]
        runs.append({"seed": sd, "effect_a9LO": lo["diff_points"], "p": lo["mcnemar_p"], "n": lo["n"], "fixes": lo["fixes"], "breaks": lo["breaks"], "fresh_acc": 100 * s["FRESH"]["acc_true"], "fresh_gap": s["FRESH"]["all"]["diff_points"],
                     "test_acc": 100 * s["TEST711"]["acc_true"], "shortest_pick": s["TEST711"]["pred_equals_shortest"], "HI_changed": s["own_strata"]["HI_changed"], "own_sizes": s["own_strata"]["sizes"],
                     "own_LO_effect": s["own_strata"]["by_stratum"].get("LO", {}).get("diff_points"), "E1_seed": bool(lo["diff_points"] >= 5.0 and lo["mcnemar_p"] < 0.0167 and 100 * s["FRESH"]["acc_true"] >= 93.5),
                     "effect_ok": bool(lo["diff_points"] >= 5.0 and lo["mcnemar_p"] < 0.0167), "acc_ok": bool(100 * s["FRESH"]["acc_true"] >= 93.5)})
    out["methods"][m] = {"runs": runs, "n_runs": len(runs), "E1": bool(len(runs) == 3 and sum(r["E1_seed"] for r in runs) >= 2)}
json.dump(out, open(ROOT + "/analysis/a24_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
