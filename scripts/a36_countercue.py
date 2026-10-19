#!/usr/bin/env python
"""A36 part 1: accuracy of the Med-CMR fine-tuned systems on counter-cue questions (correct option not the tied-shortest) against cue-following questions, FRESH3000, true and black image.
python a36_countercue.py -> analysis/a36_countercue_summary.json"""
import glob
import json
import re

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
CONTRA = {}
for c in D:
    if c.get("question type") != "mcq":
        continue
    L = [len(LAB.sub("", c[l]).strip()) for l in "ABCDE"]
    CONTRA[c["case_id"]] = L["ABCDE".index(c["answer"].strip())] != min(L)
SYSTEMS = [("A9 Qwen-3B (reference)", "ft_3b_s0_22569048_*"), ("A9 Qwen-7B (reference)", "ft_7b_s0_22569049_*")]
SYSTEMS += [(f"UL Qwen-3B seed {s % 100}", f"a26_medcmr_ul_3b_seed{s}_2*") for s in (20261041, 20261042, 20261043)]
SYSTEMS += [(f"UL Qwen-7B seed {s % 100}", f"a26_medcmr_ul_7b_seed{s}_2*") for s in (20261051, 20261052)]
SYSTEMS += [(f"LP+MU Qwen-3B seed {s % 100}", f"a2[46]_medcmr_both_3b_seed{s}_2*") for s in (20261031, 20261032, 20261033, 20261034, 20261035)]
rng = np.random.RandomState(20261104)
out = {"systems": {}}
for name, pat in SYSTEMS:
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}/rows_ft.jsonl") if "smoke" not in x]
    if not f:
        continue
    P = {}
    for ln in open(f[0]):
        r = json.loads(ln)
        if r["group"].lower() == "fresh" and r["arm"] in ("base", "black") and r["case_id"] in CONTRA:
            P.setdefault(r["arm"], {})[r["case_id"]] = int(max(range(5), key=lambda k: r["direct"]["lp"][k]) == "ABCDE".index(r["gold"]))
    ids = sorted(set(P["base"]) & set(P["black"]))
    res = {"n": len(ids)}
    for lab, sel in (("counter_cue", [i for i in ids if CONTRA[i]]), ("cue_following", [i for i in ids if not CONTRA[i]])):
        for arm in ("base", "black"):
            v = np.array([P[arm][i] for i in sel], float)
            boots = [100 * v[rng.randint(0, len(v), len(v))].mean() for _ in range(1000)]
            res[f"{lab}_{arm}"] = {"n": len(sel), "acc": round(100 * float(v.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}
    res["all_base_acc"] = round(100 * float(np.mean([P["base"][i] for i in ids])), 2)
    res["gap_follow_minus_counter_true"] = round(res["cue_following_base"]["acc"] - res["counter_cue_base"]["acc"], 2)
    out["systems"][name] = res
    print(name, json.dumps(res), flush=True)
S = out["systems"]
out["C1_both_A9_gap_ge_30"] = bool(all(S[k]["gap_follow_minus_counter_true"] >= 30.0 for k in ("A9 Qwen-3B (reference)", "A9 Qwen-7B (reference)")))
json.dump(out, open(ROOT + "/analysis/a36_countercue_summary.json", "w"), indent=1)
print(json.dumps({"C1": out["C1_both_A9_gap_ge_30"]}))
