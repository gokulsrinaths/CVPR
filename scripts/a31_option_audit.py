#!/usr/bin/env python
"""A31: what in the option text of Med-CMR identifies the correct answer, beyond character length? Text only (no image, no question for M1-M4 except the overlap feature of M4).
Train: MCQ cases of the training selection not in the 886-case held-out list; test: the 711 held-out MCQs (TEST711). python a31_option_audit.py -> analysis/a31_summary.json"""
import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
HEDGE = {"suggest", "suggests", "suggesting", "possible", "possibly", "likely", "probable", "probably", "apparent", "apparently", "may", "might", "could", "potential", "potentially", "consistent", "indicating", "indicates", "appears", "tenuous", "partial", "partially"}
NEG = re.compile(r"^(no|not|there is no|without|absent)\b", re.I)
TOK = re.compile(r"[a-z0-9]+")
rng = np.random.RandomState(20261101)

D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
test_ids = set(json.load(open(ROOT + "/data/train_pool/split_v1/test_ids.json")))
Q = []
for c in D:
    if c.get("question type") != "mcq":
        continue
    opts = [LAB.sub("", c[l]).strip() for l in "ABCDE"]
    Q.append({"id": c["case_id"], "q": c["question"], "opts": opts, "gold": "ABCDE".index(c["answer"].strip()), "test": c["case_id"] in test_ids})
tr = [x for x in Q if not x["test"]]
te = [x for x in Q if x["test"]]


def shortest_idx(opts):
    L = [len(o) for o in opts]
    return min(range(5), key=lambda k: (L[k], k)), L.count(min(L))


for x in Q:
    x["short"], x["ntie"] = shortest_idx(x["opts"])
    x["contra"] = x["gold"] != x["short"] and not (len(x["opts"][x["gold"]]) == len(x["opts"][x["short"]]))  # correct option not (tied) shortest
print("train", len(tr), "test", len(te), "test cue-contradicting", sum(x["contra"] for x in te), "train cue-contradicting", sum(x["contra"] for x in tr), flush=True)


def toks(s):
    return TOK.findall(s.lower())


def feats_len(x):
    L = np.array([len(o) for o in x["opts"]], float)
    W = np.array([len(toks(o)) for o in x["opts"]], float)
    rank = np.argsort(np.argsort(L))
    return np.stack([L - L.mean(), W - W.mean(), rank - 2.0, (L - L.mean()) / (L.std() + 1e-6)], 1)


def feats_style(x):
    qt = set(toks(x["q"]))
    T = [toks(o) for o in x["opts"]]
    rows = []
    for k, o in enumerate(x["opts"]):
        t = T[k] or ["_"]
        hedge = sum(w in HEDGE for w in t) / len(t)
        comma = o.count(",") / len(t)
        overlap_q = sum(w in qt for w in t) / len(t)
        others = set(w for j in range(5) if j != k for w in T[j])
        cons = sum(w in others for w in t) / len(t)
        jac = np.mean([len(set(t) & set(T[j])) / max(1, len(set(t) | set(T[j]))) for j in range(5) if j != k])
        rows.append([hedge, comma, overlap_q, cons, jac, float(bool(NEG.match(o))), float(any(ch.isdigit() for ch in o))])
    M = np.array(rows, float)
    return M - M.mean(0)


def option_rows(items, feat=None):
    X, y, g = [], [], []
    for qi, x in enumerate(items):
        F = feat(x) if feat else None
        for k in range(5):
            X.append(F[k] if F is not None else None)
            y.append(int(k == x["gold"]))
            g.append(qi)
    return (np.array(X) if feat else None), np.array(y), np.array(g)


def texts(items):
    return [o for x in items for o in x["opts"]]


def evaluate(scores, items, subset):
    s = scores.reshape(-1, 5)
    pred = s.argmax(1)
    idx = [i for i, x in enumerate(items) if subset(x)]
    ok = np.array([pred[i] == items[i]["gold"] for i in idx], float)
    pk = np.array([pred[i] == items[i]["short"] for i in idx], float)
    boot = [rng.randint(0, len(ok), len(ok)) for _ in range(1000)]
    lo, hi = np.percentile([100 * ok[b].mean() for b in boot], [2.5, 97.5])
    return {"n": len(idx), "acc": round(100 * ok.mean(), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "picks_shortest": round(100 * pk.mean(), 2)}


def chance(items, subset):
    return round(100 * float(np.mean([1.0 / (5 - 1) if x["contra"] else 0.2 for x in items if subset(x)])), 1)


ALL = lambda x: True
CON = lambda x: x["contra"]
out = {"n_train": len(tr), "n_test": len(te), "n_test_contra": sum(x["contra"] for x in te), "n_train_contra": sum(x["contra"] for x in tr), "models": {}, "chance_contra_pct": 25.0}


def fit_predict(name, train_items, feat_kind):
    ytr = option_rows(train_items)[1]
    if feat_kind == "len":
        Xtr, Xte = [np.vstack([feats_len(x) for x in it]) for it in (train_items, te)]
    elif feat_kind == "style":
        Xtr, Xte = [np.vstack([feats_style(x) for x in it]) for it in (train_items, te)]
    elif feat_kind == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vec.fit_transform(texts(train_items))
        Xte = vec.transform(texts(te))
    elif feat_kind == "tfidf+style":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        A = vec.fit_transform(texts(train_items))
        B = vec.transform(texts(te))
        Xtr = hstack([A, csr_matrix(np.vstack([feats_style(x) for x in train_items]))]).tocsr()
        Xte = hstack([B, csr_matrix(np.vstack([feats_style(x) for x in te]))]).tocsr()
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    if feat_kind in ("len", "style"):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf.fit(Xtr, ytr)
    sc = clf.decision_function(Xte)
    out["models"][name] = {"test_all": evaluate(sc, te, ALL), "test_contra": evaluate(sc, te, CON)}
    print(name, json.dumps(out["models"][name]), flush=True)


# M0: the shortest-option rule (ties to the first option)
sc0 = np.array([[-len(o) - 1e-3 * k for k, o in enumerate(x["opts"])] for x in te]).reshape(-1)
out["models"]["M0_shortest_rule"] = {"test_all": evaluate(sc0, te, ALL), "test_contra": evaluate(sc0, te, CON)}
print("M0", json.dumps(out["models"]["M0_shortest_rule"]), flush=True)
trc = [x for x in tr if x["contra"]]
fit_predict("M1_length_features_all_train", tr, "len")
fit_predict("M2_tfidf_all_train", tr, "tfidf")
fit_predict("M3_tfidf_contra_train_only", trc, "tfidf")
fit_predict("M4_style_nonlength_all_train", tr, "style")
fit_predict("M5_style_nonlength_contra_train_only", trc, "style")
fit_predict("M6_tfidf_plus_style_contra_train_only", trc, "tfidf+style")
# rules (A31)
mm = out["models"]
out["W1_wording_beyond_length"] = bool(max(mm[k]["test_contra"]["acc"] for k in ("M3_tfidf_contra_train_only",)) >= 35.0 and mm["M3_tfidf_contra_train_only"]["test_contra"]["ci95"][0] >= 28.0)
out["W2_nonlength_style_enough"] = bool(mm["M4_style_nonlength_all_train"]["test_all"]["acc"] >= 60.0)
json.dump(out, open(ROOT + "/analysis/a31_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
