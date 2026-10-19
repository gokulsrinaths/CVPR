#!/usr/bin/env python
"""A32: a model-independent stratifier. Text-only option classifier (word + bigram TF-IDF, option-level logistic regression, no image, no question), 5-fold out-of-fold scores (folds grouped by near-duplicate component) over all
non-held-out Med-CMR MCQs; stratify FRESH3000 by its confidence (top probability of the within-question softmax over the five option scores) in terciles; compare with the black-image strata of
the fine-tuned models. python a32_stratifier.py -> analysis/a32_summary.json"""
import glob
import json
import re

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
test_ids = set(json.load(open(ROOT + "/data/train_pool/split_v1/test_ids.json")))
Q = [{"id": c["case_id"], "opts": [LAB.sub("", c[l]).strip() for l in "ABCDE"], "gold": "ABCDE".index(c["answer"].strip())} for c in D if c.get("question type") == "mcq"]
pool = [x for x in Q if x["id"] not in test_ids]
print("pool", len(pool), flush=True)
ptop = {}
COMP = json.load(open(ROOT + "/data/train_pool/split_v1/components.json"))
groups = [COMP.get(x["id"], x["id"]) for x in pool]
kf = GroupKFold(5)
for tri, tei in kf.split(pool, groups=groups):
    tr, te = [pool[i] for i in tri], [pool[i] for i in tei]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform([o for x in tr for o in x["opts"]])
    ytr = np.array([int(k == x["gold"]) for x in tr for k in range(5)])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    s = clf.decision_function(vec.transform([o for x in te for o in x["opts"]])).reshape(-1, 5)
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for x, pi in zip(te, p):
        ptop[x["id"]] = float(pi.max())
RUNS = {"A9 Qwen-3B": "ft_3b_s0_22569048_*", "A9 Qwen-7B": "ft_7b_s0_22569049_*"}


def lsm(l):
    m = max(l)
    z = m + np.log(sum(np.exp(x - m) for x in l))
    return [x - z for x in l]


out = {"runs": {}}
for name, pat in RUNS.items():
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}/rows_ft.jsonl")]
    P = {}
    for ln in open(f[0]):
        r = json.loads(ln)
        if r["group"] == "fresh":
            lp = lsm(r["direct"]["lp"])
            P.setdefault(r["case_id"], {})[r["arm"]] = (int(np.argmax(lp)), r["gold"], sorted(np.exp(lp))[-1] - sorted(np.exp(lp))[-2])
    ids = [i for i in P if i in ptop and all(a in P[i] for a in ("base", "black", "mismatch_rand", "mismatch_nn"))]
    gold = {i: "ABCDE".index(P[i]["base"][1]) for i in ids}
    pt = np.array([ptop[i] for i in ids])
    mb = np.array([P[i]["black"][2] for i in ids])
    rho = float(spearmanr(pt, mb)[0])
    q1, q2 = np.percentile(pt, [100 / 3, 200 / 3])
    strata = {"bottom": [i for i, v in zip(ids, pt) if v <= q1], "middle": [i for i, v in zip(ids, pt) if q1 < v <= q2], "top": [i for i, v in zip(ids, pt) if v > q2]}
    res = {"n": len(ids), "spearman_ptext_vs_black_margin": round(rho, 3), "strata": {}}
    for sname, sel in strata.items():
        a = lambda arm: 100 * float(np.mean([P[i][arm][0] == gold[i] for i in sel]))
        fixes = sum(P[i]["base"][0] == gold[i] and P[i]["black"][0] != gold[i] for i in sel)
        breaks = sum(P[i]["base"][0] != gold[i] and P[i]["black"][0] == gold[i] for i in sel)
        p_mc = float(binomtest(min(fixes, breaks), fixes + breaks, 0.5).pvalue) if fixes + breaks else 1.0
        chg = lambda arm: round(100 * float(np.mean([P[i]["base"][0] != P[i][arm][0] for i in sel])), 2)
        res["strata"][sname] = {"n": len(sel), "acc_true": round(a("base"), 2), "acc_black": round(a("black"), 2), "g": round(a("base") - a("black"), 2), "fixes": int(fixes), "breaks": int(breaks), "mcnemar_p": p_mc,
                                "pct_changed_vs_black": chg("black"), "pct_changed_vs_random": chg("mismatch_rand"), "pct_changed_vs_plausible_wrong": chg("mismatch_nn"), "mean_ptext": round(float(np.mean([ptop[i] for i in sel])), 3)}
    res["S1_top_tertile_changed_le_2pct"] = bool(max(res["strata"]["top"][k] for k in ("pct_changed_vs_black", "pct_changed_vs_random", "pct_changed_vs_plausible_wrong")) <= 2.0)
    res["S3_agreement_rho_ge_0.4"] = bool(rho >= 0.4)
    out["runs"][name] = res
    print(name, json.dumps(res), flush=True)
out["S1_both"] = bool(all(v["S1_top_tertile_changed_le_2pct"] for v in out["runs"].values()))
out["S3_both"] = bool(all(v["S3_agreement_rho_ge_0.4"] for v in out["runs"].values()))
json.dump(out, open(ROOT + "/analysis/a32_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
