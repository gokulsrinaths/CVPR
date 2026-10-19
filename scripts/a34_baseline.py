#!/usr/bin/env python
"""A34: existing sample-level measures against the stratified measures on the same fine-tuned systems. python a34_baseline.py -> analysis/a34_summary.json"""
import glob
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SYSTEMS = [("Med-CMR Qwen-3B", "ft_3b_s0_22569048_*", "fresh", "val700"), ("Med-CMR Qwen-7B", "ft_7b_s0_22569049_*", "fresh", "val700"),
           ("PMC-VQA Qwen-3B", "pmc_q3b_s0_*", "E2", "VAL700"), ("PMC-VQA Qwen-7B", "pmc_q7b_s0_*", "E2", "VAL700"), ("PMC-VQA InternVL3-2B", "pmc_ivl2b_s0_*", "E2", "VAL700"),
           ("OmniMedVQA Qwen-3B", "omni16_q3b_seed20261003_*", "E2", "VAL700"), ("OmniMedVQA Qwen-7B", "omni16_q7b_seed20261003_*", "E2", "VAL700"), ("OmniMedVQA InternVL3-2B", "omni16_ivl2b_seed20261003_*", "E2", "VAL700")]


def lsm(l):
    l = np.array(l, float)
    m = l.max()
    return l - (m + np.log(np.exp(l - m).sum()))


out = {"systems": {}}
for name, pat, grp, vgrp in SYSTEMS:
    d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}") if "smoke" not in x and glob.glob(x + "/rows_ft.jsonl")][0]
    T, B, V = {}, {}, []
    for ln in open(d + "/rows_ft.jsonl"):
        r = json.loads(ln)
        if r["group"] == grp and r["arm"] in ("base", "black"):
            (T if r["arm"] == "base" else B)[r["case_id"]] = (lsm(r["direct"]["lp"]), r["gold"])
        if r["group"] == vgrp and r["arm"] == "base":
            p = np.sort(np.exp(lsm(r["direct"]["lp"])))[::-1]
            V.append(p[0] - p[1])
    tau = float(np.sort(V)[int(0.15 * len(V))])
    ids = sorted(set(T) & set(B))
    gi = {i: "ABCDE".index(T[i][1]) for i in ids}
    acc_t = np.array([int(T[i][0].argmax() == gi[i]) for i in ids])
    acc_b = np.array([int(B[i][0].argmax() == gi[i]) for i in ids])
    vig = np.array([T[i][0][gi[i]] - B[i][0][gi[i]] for i in ids])
    chg = np.array([int(T[i][0].argmax() != B[i][0].argmax()) for i in ids])
    mb = np.array([(lambda p: p[0] - p[1])(np.sort(np.exp(B[i][0]))[::-1]) for i in ids])
    hi, lo = mb >= tau, mb < 0.90
    out["systems"][name] = {"n": len(ids), "acc_drop_points": round(100 * float(acc_t.mean() - acc_b.mean()), 2), "mean_VIG_nats": round(float(vig.mean()), 3), "median_VIG_nats": round(float(np.median(vig)), 3),
                            "pct_answers_changed_by_black_image": round(100 * float(chg.mean()), 2), "HI_share_pct": round(100 * float(hi.mean()), 1), "share_black_margin_ge_099_pct": round(100 * float((mb >= 0.99).mean()), 1), "pct_changed_in_margin_ge_099": round(100 * float(chg[mb >= 0.99].mean()), 2) if (mb >= 0.99).sum() else None, "pct_changed_in_HI": round(100 * float(chg[hi].mean()), 2) if hi.sum() else None,
                            "LO_n": int(lo.sum()), "LO_image_effect_points": round(100 * float(acc_t[lo].mean() - acc_b[lo].mean()), 2) if lo.sum() else None}
    print(name, json.dumps(out["systems"][name]), flush=True)
S = out["systems"]
drops = [S[k]["acc_drop_points"] for k in S]
vigs = [S[k]["mean_VIG_nats"] for k in S]
chg = [S[k]["pct_answers_changed_by_black_image"] for k in S]
his = [S[k]["HI_share_pct"] for k in S]
out["spearman_drop_vs_VIG"] = round(float(spearmanr(drops, vigs)[0]), 3)
out["spearman_drop_vs_changed"] = round(float(spearmanr(drops, chg)[0]), 3)
out["spearman_drop_vs_HI_share"] = round(float(spearmanr(drops, his)[0]), 3)
out["R1_conventional_measures_equivalent"] = bool(out["spearman_drop_vs_VIG"] >= 0.9 and out["spearman_drop_vs_changed"] >= 0.9)
json.dump(out, open(ROOT + "/analysis/a34_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "systems"}))
