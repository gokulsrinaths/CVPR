#!/usr/bin/env python
"""A41 analysis: exactly rules Y1, Y2 and the reading rule of docs/preregistration_v1_amendment_A41.md. python a41_analysis.py -> analysis/a41_summary.json"""
import glob
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
BL = json.load(open(ROOT + "/analysis/a36_blind_summary.json"))["benchmarks"]
BENCH = ["Med-CMR", "PMC-VQA", "OmniMedVQA", "A-OKVQA", "MedXpertQA-MM", "MMMU health"]
rng = np.random.RandomState(20261111)
f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a41_3b_2*/rows.jsonl") if "smoke" not in x and sum(1 for _ in open(x)) == 24000][0]
A, chance = {}, {}
for ln in open(f):
    r = json.loads(ln)
    A.setdefault((r["bench"], r["arm"]), []).append(int(int(np.argmax(r["lp"])) == r["gold"]))
    chance.setdefault(r["bench"], []).append(1.0 / len(r["lp"]))
res = {}
for b in BENCH:
    row = {"n": len(A[(b, "true")]), "chance": round(100 * float(np.mean(chance[b])), 1), "blind_ceiling": BL[b]["B_opt"]}
    for arm in ("true", "black", "told"):
        v = np.array(A[(b, arm)], float)
        boots = [100 * v[rng.randint(0, len(v), len(v))].mean() for _ in range(1000)]
        row[arm] = {"acc": round(100 * float(v.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}
    row["true_minus_black"] = round(row["true"]["acc"] - row["black"]["acc"], 2)
    row["black_minus_chance"] = round(row["black"]["acc"] - row["chance"], 2)
    row["ceiling_minus_black"] = round(row["blind_ceiling"] - row["black"]["acc"], 2)
    res[b] = row
    print(b, json.dumps(row), flush=True)
# fine-tuned Qwen-3B black-image accuracies on the same sets
FT = {}
d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/ft_3b_s0_22569048_*/rows_ft.jsonl")][0]
v = [int(max(range(5), key=lambda k: json.loads(l)["direct"]["lp"][k]) == "ABCDE".index(json.loads(l)["gold"])) for l in open(d) if json.loads(l)["group"] == "test711" and json.loads(l)["arm"] == "black"]
FT["Med-CMR"] = round(100 * float(np.mean(v)), 2)
for name, pat, grp in (("OmniMedVQA", "omni16_q3b_seed20261003_*", "E1"), ("PMC-VQA", "pmc_q3b_s0_*", "E1")):
    d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}/rows_ft.jsonl")][0]
    v = []
    for l in open(d):
        r = json.loads(l)
        if r["group"] == grp and r["arm"] == "black":
            v.append(int(max(range(4), key=lambda k: r["direct"]["lp"][k]) == "ABCD".index(r["gold"])))
    FT[name] = round(100 * float(np.mean(v)), 2)
d = glob.glob(f"{ROOT}/experiments/pilot/a21_q3b_length0.25_seed20261021_2*/summary_dose.json")[0]
FT["A-OKVQA"] = json.load(open(d))["orig_E1"]["acc_black"]
for b in FT:
    res[b]["finetuned_3b_black"] = FT[b]
    res[b]["finetuned_minus_zeroshot_black"] = round(FT[b] - res[b]["black"]["acc"], 2)
out = {"model": "Qwen2.5-VL-3B-Instruct zero-shot", "benchmarks": res}
out["Y1_artifact_route"] = bool(res["Med-CMR"]["ceiling_minus_black"] >= 30.0 and res["OmniMedVQA"]["ceiling_minus_black"] >= 30.0)
out["Y2_finetuning_installs"] = bool(res["Med-CMR"]["finetuned_minus_zeroshot_black"] >= 40.0 and res["OmniMedVQA"]["finetuned_minus_zeroshot_black"] >= 20.0)
xs = [res[b]["blind_ceiling"] - res[b]["chance"] for b in BENCH]
ys = [res[b]["black_minus_chance"] for b in BENCH]
out["spearman_ceiling_excess_vs_black_excess"] = round(float(spearmanr(xs, ys)[0]), 3)
json.dump(out, open(ROOT + "/analysis/a41_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "benchmarks"}))
