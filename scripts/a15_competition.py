#!/usr/bin/env python
"""A15 Part 1: text-vision competition score for predicting visual overrides. python a15_competition.py -> analysis/a15_competition_summary.json"""
import glob
import json
import math
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
P = ROOT + "/experiments/pilot/"
SYS = [
    ("Med-CMR", "Released adapter", P + "condrel_fresh_s0_22567079_*/rows.jsonl", "fresh", 5),
    ("Med-CMR", "Qwen 3B", P + "ft_3b_s0_22569048_*/rows_ft.jsonl", "fresh", 5),
    ("Med-CMR", "Qwen 7B", P + "ft_7b_s0_22569049_*/rows_ft.jsonl", "fresh", 5),
    ("Med-CMR", "InternVL3-2B", P + "ft3_ivl2b_s0_22584009_*/rows_ft.jsonl", "fresh", 5),
    ("PMC-VQA", "Qwen 3B", P + "pmc_q3b_s0_22584006_*/rows_ft.jsonl", "E1", 4),
    ("PMC-VQA", "Qwen 7B", P + "pmc_q7b_s0_22584007_*/rows_ft.jsonl", "E1", 4),
    ("PMC-VQA", "InternVL3-2B", P + "pmc_ivl2b_s0_22584008_*/rows_ft.jsonl", "E1", 4),
    ("Planted", "Qwen 3B", P + "a15pmc_q3b_implant_2*/rows_a15.jsonl", "E1", 4),
    ("Planted", "Qwen 7B", P + "a15pmc_q7b_implant_2*/rows_a15.jsonl", "E1", 4),
    ("Planted", "InternVL3-2B", P + "a15pmc_ivl2b_implant_2*/rows_a15.jsonl", "E1", 4),
]
EPS = 0.1
LET = "ABCDE"


def lsm(l):
    m = max(l)
    z = m + math.log(sum(math.exp(x - m) for x in l))
    return [x - z for x in l]


def load(pat, grp, K):
    f = [x for x in glob.glob(pat) if "smoke" not in x]
    if not f:
        return None
    t = {}
    for x in open(sorted(f)[-1]):
        r = json.loads(x)
        if r.get("group") == grp:
            t.setdefault(r["case_id"], {})[r["arm"]] = r
    rows = []
    for cid, a in sorted(t.items()):
        if not all(k in a for k in ("base", "black", "mismatch_rand", "mismatch_nn")):
            continue
        l = {k: lsm(a[k]["direct"]["lp"][:K]) for k in ("base", "black", "mismatch_rand", "mismatch_nn")}
        ab = max(range(K), key=lambda i: l["black"][i])
        mt = l["black"][ab] - max(l["black"][i] for i in range(K) if i != ab)

        def delta(arm):
            v = max(l[arm][i] for i in range(K) if i != ab) - l[arm][ab]
            return mt + v
        dw = max(delta("mismatch_rand"), delta("mismatch_nn"))
        pb = sorted([math.exp(x) for x in l["black"]], reverse=True)
        o = int(max(range(K), key=lambda i: l["base"][i]) != ab)
        rows.append((cid, mt, pb[0] - pb[1], dw, o))
    return rows


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    srt = allv[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


D = {}
for ds, name, pat, grp, K in SYS:
    r = load(pat, grp, K)
    if r is None:
        print("MISSING", ds, name)
        continue
    a = np.array([[x[1], x[2], x[3], x[4]] for x in r])
    D[(ds, name)] = {"mt": a[:, 0], "mb": a[:, 1], "dw": a[:, 2], "o": a[:, 3].astype(int)}
out = {"systems": {}}
rng = np.random.RandomState(20261004)
for (ds, name), d in D.items():
    mt, mb, dw, o = d["mt"], d["mb"], d["dw"], d["o"]
    R = dw / (mt + EPS)
    S = {"S1_neg_prob_margin": -mb, "S2_neg_logodds_margin": -mt, "S3_visual_alone": dw, "S4_competition_ratio": R, "S5_deficit": dw - mt}
    res = {"n": int(len(o)), "overrides": int(o.sum()), "override_rate": round(float(o.mean()), 4), "median_Mt": round(float(np.median(mt)), 3), "median_R": round(float(np.median(R)), 3),
           "auroc": {k: (round(auroc(v, o), 4) if auroc(v, o) is not None else None) for k, v in S.items()},
           "auprc": {k: round(float(average_precision_score(o, v)), 4) for k, v in S.items()}}
    if o.sum() >= 40:
        base = max(res["auroc"]["S1_neg_prob_margin"], res["auroc"]["S2_neg_logodds_margin"])
        dd = []
        n = len(o)
        for _ in range(500):
            i = rng.randint(0, n, n)
            if o[i].sum() == 0 or o[i].sum() == n:
                continue
            dd.append(auroc(S["S4_competition_ratio"][i], o[i]) - max(auroc(S["S1_neg_prob_margin"][i], o[i]), auroc(S["S2_neg_logodds_margin"][i], o[i])))
        res["delta_auroc_S4_vs_best_text"] = round(res["auroc"]["S4_competition_ratio"] - base, 4)
        res["delta_ci95"] = [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]
    # binned P(O) against log10 R and against prob margin (collapse plot data)
    xr = np.log10(np.maximum(R, 1e-3))
    bins = [(-9, -2), (-2, -1), (-1, 0), (0, 1), (1, 9)]
    res["P_override_by_logR"] = [[b[0], b[1], int(((xr >= b[0]) & (xr < b[1])).sum()), round(float(o[(xr >= b[0]) & (xr < b[1])].mean()), 4) if ((xr >= b[0]) & (xr < b[1])).sum() else None] for b in bins]
    mbins = [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    res["P_override_by_prob_margin"] = [[b[0], b[1], int(((mb >= b[0]) & (mb < b[1])).sum()), round(float(o[(mb >= b[0]) & (mb < b[1])].mean()), 4) if ((mb >= b[0]) & (mb < b[1])).sum() else None] for b in mbins]
    out["systems"][f"{ds}|{name}"] = res
    print(ds, name, res["n"], res["overrides"], res["auroc"], res.get("delta_auroc_S4_vs_best_text"), res.get("delta_ci95"), flush=True)

# leave-one-system-out transfer
keys = list(D)
loso = {}
for j in keys:
    Xt, Xr, Xb, Y = [], [], [], []
    for k in keys:
        if k == j:
            continue
        d = D[k]
        idx = list(range(len(d["o"])))
        random.Random(20261004).shuffle(idx)
        idx = idx[:1500]
        mt, dw, o = d["mt"][idx], d["dw"][idx], d["o"][idx]
        Xt += list(mt)
        Xr += list(np.clip(np.log10(np.maximum(dw / (mt + EPS), 1e-3)), -3, 3))
        Xb += list(zip(mt, np.log10(np.maximum(dw, 1e-3))))
        Y += list(o)
    d = D[j]
    mt, dw, o = d["mt"], d["dw"], d["o"]
    xr = np.clip(np.log10(np.maximum(dw / (mt + EPS), 1e-3)), -3, 3)
    res = {}
    for nm, xtr, xte in (("text_only_Mt", np.array(Xt).reshape(-1, 1), mt.reshape(-1, 1)), ("competition_logR", np.array(Xr).reshape(-1, 1), xr.reshape(-1, 1)),
                         ("additive_Mt_logDelta", np.array(Xb), np.column_stack([mt, np.log10(np.maximum(dw, 1e-3))]))):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, Y)
        p = np.clip(m.predict_proba(xte)[:, 1], 1e-6, 1 - 1e-6)
        res[nm] = {"logloss": round(float(log_loss(o, p, labels=[0, 1])), 4), "ece": round(ece(p, o), 4), "mean_p": round(float(p.mean()), 4), "base_rate": round(float(o.mean()), 4)}
    out["systems"][f"{j[0]}|{j[1]}"]["loso"] = res
    print("LOSO", j, {k: v["logloss"] for k, v in res.items()}, flush=True)

ev = [k for k in out["systems"] if out["systems"][k]["overrides"] >= 40 and "delta_ci95" in out["systems"][k]]
c1 = [out["systems"][k]["delta_auroc_S4_vs_best_text"] >= 0.02 and out["systems"][k]["delta_ci95"][0] > 0 for k in ev]
lo = [k for k in out["systems"] if "loso" in out["systems"][k]]
wins = [out["systems"][k]["loso"]["competition_logR"]["logloss"] < out["systems"][k]["loso"]["text_only_Mt"]["logloss"] for k in lo]
rel = [(out["systems"][k]["loso"]["text_only_Mt"]["logloss"] - out["systems"][k]["loso"]["competition_logR"]["logloss"]) / out["systems"][k]["loso"]["text_only_Mt"]["logloss"] for k in lo]
out["verdicts"] = {"n_systems": len(D), "complete_10": len(D) == 10,
                   "C1_evaluable": len(ev), "C1_satisfied": int(sum(c1)), "C1_holds": bool(len(ev) and sum(c1) / len(ev) >= 0.7),
                   "C2_wins": int(sum(wins)), "C2_of": len(lo), "C2_mean_rel_reduction": round(float(np.mean(rel)), 4), "C2_holds": bool(len(D) == 10 and sum(wins) >= 8 and np.mean(rel) >= 0.05)}
c3 = {}
for fam in ("Qwen 3B", "Qwen 7B", "InternVL3-2B"):
    po, pl = out["systems"].get("PMC-VQA|" + fam), out["systems"].get("Planted|" + fam)
    c3[fam] = bool(po and pl and pl["median_Mt"] > po["median_Mt"] and pl["median_R"] < po["median_R"])
out["verdicts"]["C3_mechanism"] = c3
print(out["verdicts"])
json.dump(out, open(ROOT + "/analysis/a15_competition_summary.json", "w"), indent=1)
