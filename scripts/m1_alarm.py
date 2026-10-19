#!/usr/bin/env python
"""A12 M1: counterfactual-evidence alarm. Compares the margin alarm with a logistic combiner of confidence and image-evidence features.
python m1_alarm.py   (reads the frozen run directories; writes analysis/m1_alarm_summary.json)"""
import glob
import hashlib
import json
import math
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
RUNS = {
    ("PMC-VQA", "Qwen2.5-VL-3B"): ("pmc_q3b_s0_22584006_*", ("E2", "E1"), 4),
    ("PMC-VQA", "Qwen2.5-VL-7B"): ("pmc_q7b_s0_22584007_*", ("E2", "E1"), 4),
    ("PMC-VQA", "InternVL3-2B"): ("pmc_ivl2b_s0_22584008_*", ("E2", "E1"), 4),
    ("Med-CMR", "Qwen2.5-VL-3B"): ("ft_3b_s0_22569048_*", ("fresh", None), 5),
    ("Med-CMR", "Qwen2.5-VL-7B"): ("ft_7b_s0_22569049_*", ("fresh", None), 5),
    ("Med-CMR", "InternVL3-2B"): ("ft3_ivl2b_s0_22584009_*", ("fresh", None), 5),
}
FLAG = 0.15


def softmax(l):
    m = max(l)
    e = [math.exp(x - m) for x in l]
    s = sum(e)
    return [x / s for x in e]


def js(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda a, b: sum(x * math.log(x / y) for x, y in zip(a, b) if x > 1e-12)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def feats(rows, group, K):
    t = {}
    for r in rows:
        if r["group"] == group:
            t.setdefault(r["case_id"], {})[r["arm"]] = r
    X, y, ids = [], [], []
    for cid, a in sorted(t.items()):
        if not all(k in a for k in ("base", "black", "mismatch_rand", "mismatch_nn")):
            continue
        P = {k: softmax(a[k]["direct"]["lp"][:K]) for k in a}
        am = {k: max(range(K), key=lambda i: P[k][i]) for k in P}
        srt = sorted(P["base"], reverse=True)
        srb = sorted(P["black"], reverse=True)
        gold = a["base"]["gold"]
        gi = "ABCDE".index(gold)
        X.append([srt[0] - srt[1], srb[0] - srb[1], js(P["base"], P["black"]), float(am["base"] == am["mismatch_rand"]), float(am["base"] == am["mismatch_nn"]),
                  js(P["base"], P["mismatch_rand"]), js(P["base"], P["mismatch_nn"]), srt[0]])
        y.append(int(am["base"] != gi))
        ids.append(cid)
    return np.array(X), np.array(y), ids


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])
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


def capture(score, y):
    n = len(y)
    k = int(round(FLAG * n))
    idx = np.argsort(-score, kind="mergesort")[:k]
    return float(y[idx].sum() / max(1, y.sum()))


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
    return lr.decision_function(sc.transform(Xte))


def boot_delta(s1, s2, y, B=1000, seed=20261004):
    rng = np.random.RandomState(seed)
    n, d = len(y), []
    for _ in range(B):
        i = rng.randint(0, n, n)
        a, b = auroc(s1[i], y[i]), auroc(s2[i], y[i])
        if a is not None and b is not None:
            d.append(a - b)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


out = {}
for (ds, mdl), (pat, (g_fit, g_test), K) in RUNS.items():
    d = glob.glob(f"{ROOT}/experiments/pilot/{pat}")[0]
    rows = [json.loads(x) for x in open(d + "/rows_ft.jsonl")]
    if ds == "PMC-VQA":
        Xa, ya, _ = feats(rows, g_fit, K)
        Xb, yb, _ = feats(rows, g_test, K)
        sc_comb = fit_predict(Xa, ya, Xb)
        te_y, te_x, comb = yb, Xb, sc_comb
    else:
        X, y, ids = feats(rows, g_fit, K)
        par = np.array([int(hashlib.sha256(("A12cv" + c).encode()).hexdigest()[:8], 16) % 2 for c in ids])
        comb = np.zeros(len(y))
        for f in (0, 1):
            comb[par == f] = fit_predict(X[par != f], y[par != f], X[par == f])
        te_y, te_x = y, X
    base = -te_x[:, 0]
    r = {"n": int(len(te_y)), "errors": int(te_y.sum()), "auroc_margin": round(auroc(base, te_y), 4), "auroc_combined": round(auroc(comb, te_y), 4),
         "capture15_margin": round(capture(base, te_y), 4), "capture15_combined": round(capture(comb, te_y), 4)}
    r["delta_auroc"] = round(r["auroc_combined"] - r["auroc_margin"], 4)
    r["delta_auroc_ci95"] = boot_delta(comb, base, te_y)
    r["delta_capture15_points"] = round(100 * (r["capture15_combined"] - r["capture15_margin"]), 2)
    out[f"{ds}|{mdl}"] = r
    print(ds, mdl, r, flush=True)
v = {}
pm = [out[f"PMC-VQA|{m}"] for m in ("Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "InternVL3-2B")]
mc = [out[f"Med-CMR|{m}"] for m in ("Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "InternVL3-2B")]
v["M1a_auroc_gain_pmc"] = all(r["delta_auroc"] >= 0.03 for r in pm)
v["M1b_capture_gain_pmc"] = all(r["delta_capture15_points"] >= 3.0 for r in pm)
v["M1c_no_harm_medcmr"] = all(r["delta_auroc"] >= -0.01 for r in mc)
out["verdicts"] = v
print(v)
json.dump(out, open(ROOT + "/analysis/m1_alarm_summary.json", "w"), indent=1)
