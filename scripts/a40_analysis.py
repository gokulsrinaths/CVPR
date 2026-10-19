#!/usr/bin/env python
"""A40 analysis: exactly rules Z1-Z3 and the reading rule of docs/preregistration_v1_amendment_A40.md. python a40_analysis.py -> analysis/a40_summary.json"""
import glob
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
BL = json.load(open(ROOT + "/analysis/a36_blind_summary.json"))["benchmarks"]
BENCH = ["Med-CMR", "PMC-VQA", "OmniMedVQA", "A-OKVQA", "MedXpertQA-MM", "MMMU health"]
rng = np.random.RandomState(20261110)
out = {"models": {}}
for m in ("7b", "32b"):
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a40_{m}_2*/rows.jsonl") if "smoke" not in x]
    if not f:
        continue
    A = {}
    chance = {}
    for ln in open(f[0]):
        r = json.loads(ln)
        A.setdefault((r["bench"], r["arm"]), []).append(int(int(np.argmax(r["lp"])) == r["gold"]))
        chance.setdefault(r["bench"], []).append(1.0 / len(r["lp"]))
    res = {}
    for b in BENCH:
        row = {"n": len(A[(b, "true")]), "chance": round(100 * float(np.mean(chance[b])), 1)}
        for arm in ("true", "black", "told"):
            v = np.array(A[(b, arm)], float)
            boots = [100 * v[rng.randint(0, len(v), len(v))].mean() for _ in range(1000)]
            row[arm] = {"acc": round(100 * float(v.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}
        row["blind_ceiling"] = BL[b]["B_opt"]
        row["true_minus_black"] = round(row["true"]["acc"] - row["black"]["acc"], 2)
        row["told_minus_black"] = round(row["told"]["acc"] - row["black"]["acc"], 2)
        row["black_minus_chance"] = round(row["black"]["acc"] - row["chance"], 2)
        row["ceiling_minus_black"] = round(row["blind_ceiling"] - row["black"]["acc"], 2)
        res[b] = row
        print(m, b, json.dumps(row), flush=True)
    out["models"][m] = res
    xs = [res[b]["blind_ceiling"] - res[b]["chance"] for b in BENCH]
    ys = [res[b]["black_minus_chance"] for b in BENCH]
    out["models"][m]["spearman_ceiling_excess_vs_black_excess"] = round(float(spearmanr(xs, ys)[0]), 3)
if "32b" in out["models"] and "7b" in out["models"]:
    r32, r7 = out["models"]["32b"], out["models"]["7b"]
    out["Z1_knowledge_route"] = bool(sum(r32[b]["black_minus_chance"] >= 10.0 for b in ("PMC-VQA", "MedXpertQA-MM", "MMMU health")) >= 2)
    out["Z2_artifact_route"] = bool(all(r7[b]["ceiling_minus_black"] >= 30.0 for b in ("Med-CMR", "OmniMedVQA")))
    out["Z3_told_lower_on_at_least_4"] = bool(sum(r32[b]["told_minus_black"] < 0 for b in BENCH) >= 4)
json.dump(out, open(ROOT + "/analysis/a40_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "models"}))
