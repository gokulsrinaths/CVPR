#!/usr/bin/env python
"""A14 T: threshold sensitivity of the text-saturation conclusions (CPU only, existing rows).
python a14_threshold.py  -> analysis/a14_threshold_summary.json"""
import glob
import json
import math

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
P = ROOT + "/experiments/pilot/"
SYS = {
    ("Med-CMR", "Released adapter"): (P + "condrel_fresh_s0_22567079_*/rows.jsonl", "fresh", 5, "ABCDE"),
    ("Med-CMR", "Qwen 3B"): (P + "ft_3b_s0_22569048_*/rows_ft.jsonl", "fresh", 5, "ABCDE"),
    ("Med-CMR", "Qwen 7B"): (P + "ft_7b_s0_22569049_*/rows_ft.jsonl", "fresh", 5, "ABCDE"),
    ("Med-CMR", "InternVL3-2B"): (P + "ft3_ivl2b_s0_22584009_*/rows_ft.jsonl", "fresh", 5, "ABCDE"),
    ("PMC-VQA", "Qwen 3B"): (P + "pmc_q3b_s0_22584006_*/rows_ft.jsonl", "E1", 4, "ABCD"),
    ("PMC-VQA", "Qwen 7B"): (P + "pmc_q7b_s0_22584007_*/rows_ft.jsonl", "E1", 4, "ABCD"),
    ("PMC-VQA", "InternVL3-2B"): (P + "pmc_ivl2b_s0_22584008_*/rows_ft.jsonl", "E1", 4, "ABCD"),
}
HI_CUTS = [0.5, 0.9, 0.99, 0.999]
LO_CUTS = [0.5, 0.7, 0.9]


def softmax(l):
    m = max(l)
    e = [math.exp(x - m) for x in l]
    s = sum(e)
    return [x / s for x in e]


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n))


out = {}
for (ds, name), (pat, grp, K, LET) in SYS.items():
    f = glob.glob(pat)[0]
    t = {}
    for x in open(f):
        r = json.loads(x)
        if r.get("group") == grp:
            t.setdefault(r["case_id"], {})[r["arm"]] = r
    ids = sorted(i for i, a in t.items() if all(k in a for k in ("base", "black", "mismatch_rand", "mismatch_nn")))
    P_ = {i: {k: max(range(K), key=lambda j: t[i][k]["direct"]["lp"][j]) for k in t[i]} for i in ids}
    gold = {i: LET.index(t[i]["base"]["gold"]) for i in ids}
    mb = {i: margin(t[i]["black"]["direct"]["lp"][:K]) for i in ids}
    n = len(ids)
    res = {"n": n, "hi": {}, "lo": {}}
    for c in HI_CUTS:
        hi = [i for i in ids if mb[i] >= c]
        d = {"n": len(hi), "share": round(len(hi) / n, 4)}
        for arm in ("black", "mismatch_rand", "mismatch_nn"):
            d["changed_" + arm] = sum(P_[i]["base"] != P_[i][arm] for i in hi)
        d["changed_black_share"] = round(d["changed_black"] / len(hi), 4) if hi else None
        res["hi"][str(c)] = d
    for c in LO_CUTS:
        lo = [i for i in ids if mb[i] < c]
        fx = sum((P_[i]["base"] == gold[i]) and (P_[i]["black"] != gold[i]) for i in lo)
        br = sum((P_[i]["base"] != gold[i]) and (P_[i]["black"] == gold[i]) for i in lo)
        d = {"n": len(lo), "share": round(len(lo) / n, 4), "effect_points": round(100 * (fx - br) / len(lo), 2) if lo else None, "fixes": fx, "breaks": br, "mcnemar_p": mcn(fx, br) if lo else None}
        res["lo"][str(c)] = d
    out[f"{ds}|{name}"] = res
    print(ds, name, {k: (v["n"], v["changed_black"]) for k, v in res["hi"].items()}, {k: (v["n"], v["effect_points"], v["mcnemar_p"]) for k, v in res["lo"].items()}, flush=True)
v = {}
mc_all = ["Released adapter", "Qwen 3B", "Qwen 7B", "InternVL3-2B"]
v["T1_no_veto_for_cutoffs_ge_0.9"] = all(
    (out[f"Med-CMR|{s}"]["hi"][str(c)]["n"] < 100) or all(out[f"Med-CMR|{s}"]["hi"][str(c)][f"changed_{a}"] / out[f"Med-CMR|{s}"]["hi"][str(c)]["n"] <= 0.02 for a in ("black", "mismatch_rand", "mismatch_nn"))
    for s in mc_all for c in (0.9, 0.99, 0.999))
v["T2_ft_LO_effect_below_5_all_cutoffs"] = all(
    (out[f"Med-CMR|{s}"]["lo"][str(c)]["n"] < 100) or not (out[f"Med-CMR|{s}"]["lo"][str(c)]["effect_points"] >= 5 and out[f"Med-CMR|{s}"]["lo"][str(c)]["mcnemar_p"] < 0.05)
    for s in ("Qwen 3B", "Qwen 7B", "InternVL3-2B") for c in LO_CUTS)
v["T3_adapter_LO_effect_ge5_at_0.9"] = bool(out["Med-CMR|Released adapter"]["lo"]["0.9"]["effect_points"] >= 5 and out["Med-CMR|Released adapter"]["lo"]["0.9"]["mcnemar_p"] < 0.05)
out["verdicts"] = v
print(v)
json.dump(out, open(ROOT + "/analysis/a14_threshold_summary.json", "w"), indent=1)
