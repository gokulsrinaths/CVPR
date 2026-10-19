#!/usr/bin/env python
"""A5 selective prediction (docs/preregistration_v1_amendment_A5.md). CPU only; reads existing run outputs. Writes analysis/selective_prediction.json"""
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, softmax, shortest_letter, gold_from_train, LETTERS

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SEED = 20261002
TAU = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateD_retrieval_s0_22468934_*/config.json")[0]))["tau_m"]
T = gold_from_train()


def cp(k, n, a=0.05):
    """Exact Clopper-Pearson interval by bisection on the binomial tail."""
    def cdf(x, p):  # P(X <= x)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
    lo = 0.0
    if k > 0:
        a_, b_ = 0.0, k / n
        for _ in range(60):
            m = (a_ + b_) / 2
            (a_, b_) = (m, b_) if (1 - cdf(k - 1, m)) < a / 2 else (a_, m)
        lo = (a_ + b_) / 2
    hi = 1.0
    if k < n:
        a_, b_ = k / n, 1.0
        for _ in range(60):
            m = (a_ + b_) / 2
            (a_, b_) = (a_, m) if cdf(k, m) < a / 2 else (m, b_)
        hi = (a_ + b_) / 2
    return [round(lo, 4), round(hi, 4)]


def auroc(s, y):
    pos = [x for x, l in zip(s, y) if l == 1]
    neg = [x for x, l in zip(s, y) if l == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(ranks[i] for i in range(len(s)) if y[i] == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(s, y, B=2000):
    rng = random.Random(SEED)
    n = len(s)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auroc([s[i] for i in idx], [y[i] for i in idx])
        if a is not None:
            vals.append(a)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def aurc(s, y, R=200):
    rng = random.Random(SEED)
    tot = 0.0
    n = len(s)
    for _ in range(R):
        order = sorted(range(n), key=lambda i: (-s[i], rng.random()))
        err = 0
        acc = 0.0
        for k, i in enumerate(order, 1):
            err += (y[i] == 0)
            acc += err / k
        tot += acc / n
    return round(tot / R, 5)


def load():
    G = {}
    ga = {"base": {}, "black": {}}
    for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
        for r in read_jsonl(pth):
            if r["arm"] in ga:
                ga[r["arm"]][r["case_id"]] = r["direct"]["lp"]
    G["TEST711"] = {"base": ga["base"], "black": ga["black"]}
    d = {"base": {}, "black": {}}
    for r in read_jsonl(glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/forward_rows.jsonl")[0]):
        d[r["arm"]][r["case_id"]] = r["direct"]["lp"]
    G["DEV700"] = d
    p = {"base": {}, "black": {}}
    for r in read_jsonl(glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*/forward_rows.jsonl")[0]):
        if r["group"] == "pool700":
            p["base"][r["case_id"]] = r["direct"]["lp"]
    G["POOL700"] = p
    return G


def scores(lp, short):
    p = softmax(lp)
    ps = sorted(p, reverse=True)
    pred = max(range(5), key=lambda i: p[i])
    ent = -sum(x * math.log(x) for x in p if x > 0)
    return {"S1_margin": ps[0] - ps[1], "S2_top1": ps[0], "S3_negentropy": -ent, "S4_shortcut": 1.0 if LETTERS[pred] == short else 0.0}, LETTERS[pred]


def analyse(name, grp):
    ids = sorted(grp["base"])
    gold = {i: T[i]["answer"].strip() for i in ids}
    short = {i: shortest_letter(T[i]) for i in ids}
    S = {k: [] for k in ("S1_margin", "S2_top1", "S3_negentropy", "S4_shortcut")}
    y, pred = [], {}
    for i in ids:
        sc, pr = scores(grp["base"][i], short[i])
        for k in S:
            S[k].append(sc[k])
        pred[i] = pr
        y.append(int(pr == gold[i]))
    rng = random.Random(SEED)
    S["S5_random"] = [rng.random() for _ in ids]
    n = len(ids)
    err = n - sum(y)
    out = {"n": n, "errors": err, "accuracy": round(sum(y) / n, 4)}
    out["auroc"] = {k: {"auroc": round(auroc(v, y), 4), "boot95": boot_ci(v, y)} for k, v in S.items()}
    out["aurc"] = {k: aurc(v, y) for k, v in S.items()}
    m = S["S1_margin"]
    rc = []
    for cov in (0.99, 0.95, 0.90, 0.84, 0.70):
        s_thr = sorted(m, reverse=True)[max(0, math.ceil(cov * n) - 1)]
        keep = [i for i in range(n) if m[i] >= s_thr]
        k = sum(y[i] for i in keep)
        rc.append({"target_coverage": cov, "actual_coverage": round(len(keep) / n, 4), "selective_accuracy": round(k / len(keep), 4), "errors_in_answered": len(keep) - k})
    out["risk_coverage_margin"] = rc
    flagged = [i for i in range(n) if m[i] < TAU]
    ans = [i for i in range(n) if m[i] >= TAU]
    ek = sum(1 for i in flagged if y[i] == 0)
    ak = sum(y[i] for i in ans)
    out["at_tau_m"] = {"tau_m": TAU, "flagged": len(flagged), "flagged_fraction": round(len(flagged) / n, 4), "errors_captured": ek, "errors_total": err,
                       "capture_fraction": round(ek / err, 4) if err else None, "capture_cp95": cp(ek, err) if err else None,
                       "answered": len(ans), "answered_correct": ak, "answered_accuracy": round(ak / len(ans), 4), "answered_accuracy_cp95": cp(ak, len(ans)),
                       "error_rate_flagged": round(ek / len(flagged), 4) if flagged else None}
    ws = [k for k, i in enumerate(ids) if short[i] != gold[i]]
    yy = [y[k] for k in ws]
    out["shortest_wrong_subset"] = {"n": len(ws), "errors": len(ws) - sum(yy),
                                    "auroc_S1": round(auroc([m[k] for k in ws], yy), 4) if 0 < sum(yy) < len(yy) else None,
                                    "boot95_S1": boot_ci([m[k] for k in ws], yy, 1000) if 0 < sum(yy) < len(yy) else None,
                                    "auroc_S4": round(auroc([S["S4_shortcut"][k] for k in ws], yy), 4) if 0 < sum(yy) < len(yy) else None}
    if grp["black"]:
        mb = []
        same = 0
        for i in ids:
            pb = softmax(grp["black"][i])
            ps = sorted(pb, reverse=True)
            mb.append(ps[0] - ps[1])
            same += (LETTERS[max(range(5), key=lambda j: pb[j])] == pred[i])
        out["image_sensitivity"] = {"share_margin_ge_tau_true": round(len(ans) / n, 4), "share_margin_ge_tau_black": round(sum(x >= TAU for x in mb) / n, 4),
                                    "median_margin_true": round(sorted(m)[n // 2], 5), "median_margin_black": round(sorted(mb)[n // 2], 5),
                                    "prediction_unchanged_under_black": round(same / n, 4)}
    return out


G = load()
res = {"tau_m": TAU, "groups": {k: analyse(k, v) for k, v in G.items()}}
v = {}
ok1 = True
for g in ("DEV700", "POOL700"):
    t = res["groups"][g]["at_tau_m"]
    c = (0.10 <= t["flagged_fraction"] <= 0.25) and (t["capture_fraction"] is not None and t["capture_fraction"] >= 0.60) and (t["answered_accuracy"] >= 0.99)
    v["P-SP1_" + g] = {"holds": bool(c), "flagged_fraction": t["flagged_fraction"], "capture_fraction": t["capture_fraction"], "answered_accuracy": t["answered_accuracy"]}
    ok1 = ok1 and c
v["P-SP1"] = "SUPPORTED" if ok1 else "FALSIFIED"
d2 = {g: round(res["groups"][g]["auroc"]["S1_margin"]["auroc"] - res["groups"][g]["auroc"]["S4_shortcut"]["auroc"], 4) for g in ("DEV700", "POOL700")}
v["P-SP2_auroc_margin_minus_shortcut"] = d2
v["P-SP2"] = "SUPPORTED" if all(x >= 0.05 for x in d2.values()) else "FALSIFIED"
a3 = res["groups"]["DEV700"]["shortest_wrong_subset"]["auroc_S1"]
v["P-SP3_dev_shortest_wrong_auroc_S1"] = a3
v["P-SP3"] = ("SUPPORTED" if a3 >= 0.80 else "FALSIFIED") if a3 is not None else "NOT EVALUABLE"
res["verdicts"] = v
json.dump(res, open(ROOT + "/analysis/selective_prediction.json", "w"), indent=2)
print(json.dumps(res, indent=1))
