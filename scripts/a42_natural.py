#!/usr/bin/env python
"""A42: a natural dose-response. Image benefit (true minus black image accuracy) of the fine-tuned models as a function of how predictable each question is from option text alone, with nothing planted.
Predictability = top probability of the within-question softmax over the option scores of an option-level classifier (word + bigram TF-IDF of the option plus four within-question length features; no question, no image),
trained on the benchmark's own training split (Med-CMR: out-of-fold). Quintiles of the predictability within each benchmark; pooled over models. python a42_natural.py -> analysis/a42_summary.json, a42_curves.json"""
import glob
import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
D = ROOT + "/data"
PILOT = ROOT + "/experiments/pilot"
rng = np.random.RandomState(20261113)
LAB = re.compile(r"^\s*[A-E]\)\s*")


def lenfeats(opts):
    L = np.array([len(o) for o in opts], float)
    W = np.array([len(o.split()) for o in opts], float)
    return np.stack([L - L.mean(), W - W.mean(), np.argsort(np.argsort(L)) - (len(opts) - 1) / 2.0, (L - L.mean()) / (L.std() + 1e-6)], 1)


def fit_scores(train, test):
    """train/test: lists of (opts, gold_idx). returns list of softmax top-probabilities for test."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xt = vec.fit_transform([o for opts, _ in train for o in opts])
    Lt = np.vstack([lenfeats(o) for o, _ in train])
    mu, sd = Lt.mean(0), Lt.std(0) + 1e-9
    y = np.array([int(k == g) for opts, g in train for k in range(len(opts))])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(hstack([Xt, csr_matrix((Lt - mu) / sd)]).tocsr(), y)
    Xe = hstack([vec.transform([o for opts, _ in test for o in opts]), csr_matrix((np.vstack([lenfeats(o) for o, _ in test]) - mu) / sd)]).tocsr()
    s = clf.decision_function(Xe)
    out, p = [], 0
    for opts, _ in test:
        z = s[p:p + len(opts)]
        e = np.exp(z - z.max())
        out.append(float((e / e.sum()).max()))
        p += len(opts)
    return out


def read_rows(path, groups, arms=("base", "black")):
    R = {}
    for ln in open(path):
        r = json.loads(ln)
        if r["group"] in groups and r["arm"] in arms:
            lp = r["direct"]["lp"]
            R.setdefault(r["case_id"], {})[r["arm"]] = int(int(np.argmax(lp)) == "ABCDE".index(r["gold"]))
    return R


def with_rows(pattern, fname="rows_ft.jsonl"):
    return sorted(x for x in glob.glob(f"{PILOT}/{pattern}/{fname}") if "smoke" not in x)


def split_items(path, groups):
    rec = {}
    for g in json.load(open(path + "/records.json")).values():
        rec.update(g)
    ids = json.load(open(path + "/ids.json"))
    return rec, ids


BENCH = {}
# Med-CMR: out-of-fold classifier over the non-held-out MCQs, evaluated on FRESH3000
cases = json.load(open(D + "/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
test_ids = set(json.load(open(D + "/train_pool/split_v1/test_ids.json")))
comp = json.load(open(D + "/train_pool/split_v1/components.json"))
Q = {c["case_id"]: ([LAB.sub("", c[l]).strip() for l in "ABCDE"], "ABCDE".index(c["answer"].strip())) for c in cases if c.get("question type") == "mcq" and c["case_id"] not in test_ids}
ids = sorted(Q)
ptop = {}
for tri, tei in GroupKFold(5).split(ids, groups=[comp.get(i, i) for i in ids]):
    sc = fit_scores([Q[ids[i]] for i in tri], [Q[ids[i]] for i in tei])
    for i, v in zip(tei, sc):
        ptop[ids[i]] = v
mods = {"A9 Qwen-3B": with_rows("ft_3b_s0_22569048_*")[0], "A9 Qwen-7B": with_rows("ft_7b_s0_22569049_*")[0]}
BENCH["Med-CMR"] = {"ptop": ptop, "models": {m: read_rows(p, {"fresh"}) for m, p in mods.items()}}

for name, split, pats in (("PMC-VQA", "pmcvqa/split_a11", ["pmc_q3b_s0_*", "pmc_q7b_s0_*", "pmc_ivl2b_s0_*"]), ("OmniMedVQA", "omnimed/split_a16", ["omni16_q3b_seed20261003_*", "omni16_q7b_seed20261003_*", "omni16_ivl2b_seed20261003_*"])):
    rec, idsd = split_items(D + "/" + split, None)
    tr = [(rec[i]["choices"], "ABCD".index(rec[i]["gold"])) for i in idsd["TR"]]
    ev = [i for g in ("E1", "E2") for i in idsd[g]]
    sc = fit_scores(tr, [(rec[i]["choices"], "ABCD".index(rec[i]["gold"])) for i in ev])
    models = {}
    for pat in pats:
        p = with_rows(pat)[0]
        models[pat.split("_")[1] if name == "PMC-VQA" else pat.split("_")[1]] = read_rows(p, {"E1", "E2"})
    BENCH[name] = {"ptop": dict(zip(ev, sc)), "models": models}

rec, idsd = split_items(D + "/aokvqa/split_a21", None)
tr = [(rec[i]["choices"], "ABCD".index(rec[i]["gold"])) for i in idsd["TR"]]
ev = list(idsd["E1"])
sc = fit_scores(tr, [(rec[i]["choices"], "ABCD".index(rec[i]["gold"])) for i in ev])
models = {}
for fam in ("q3b", "q7b", "ivl2b"):
    for seed in (20261021, 20261022):
        f = with_rows(f"a21_{fam}_length0.25_seed{seed}_2*", "rows_ft_orig.jsonl")
        if f:
            models[f"{fam}_{seed % 100}"] = read_rows(f[0], {"E1"})
BENCH["A-OKVQA"] = {"ptop": dict(zip(ev, sc)), "models": models}

out = {"benchmarks": {}}
curves = {}
for name, B in BENCH.items():
    common = sorted(set.intersection(*[set(m) for m in B["models"].values()]) & set(B["ptop"]))
    common = [i for i in common if all(set(m[i]) == {"base", "black"} for m in B["models"].values())]
    p = np.array([B["ptop"][i] for i in common])
    edges = np.percentile(p, [20, 40, 60, 80])
    q = np.digitize(p, edges, right=True)  # 0..4, 4 = most predictable
    # per-question mean benefit across models
    ben = np.mean([[m[i]["base"] - m[i]["black"] for i in common] for m in B["models"].values()], axis=0) * 100
    true_acc = np.mean([[m[i]["base"] for i in common] for m in B["models"].values()], axis=0) * 100
    bins = []
    for k in range(5):
        sel = np.where(q == k)[0]
        boots = [ben[sel[rng.randint(0, len(sel), len(sel))]].mean() for _ in range(1000)]
        bins.append({"quintile": k + 1, "n": int(len(sel)), "mean_predictability": round(float(p[sel].mean()), 3), "benefit": round(float(ben[sel].mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
                     "true_acc": round(float(true_acc[sel].mean()), 2)})
    rho = float(spearmanr([b["quintile"] for b in bins], [b["benefit"] for b in bins])[0])
    out["benchmarks"][name] = {"n": len(common), "n_models": len(B["models"]), "bins": bins, "spearman_quintile_vs_benefit": round(rho, 3), "q1_minus_q5": round(bins[0]["benefit"] - bins[4]["benefit"], 2),
                               "N1": bool(rho <= -0.8), "N2": bool(bins[0]["benefit"] - bins[4]["benefit"] >= 5.0)}
    curves[name] = bins
    print(name, json.dumps({k: v for k, v in out["benchmarks"][name].items() if k != "bins"}), [b["benefit"] for b in bins], flush=True)
out["N1_at_least_3_of_4"] = bool(sum(v["N1"] for v in out["benchmarks"].values()) >= 3)
out["N2_at_least_3_of_4"] = bool(sum(v["N2"] for v in out["benchmarks"].values()) >= 3)
json.dump(out, open(ROOT + "/analysis/a42_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "benchmarks"}))
