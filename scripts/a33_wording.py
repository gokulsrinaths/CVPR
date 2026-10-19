#!/usr/bin/env python
"""A33: the wording signal of Med-CMR at scale. Out-of-fold (5 folds grouped by near-duplicate component) option-level logistic regression over ALL Med-CMR MCQs; accuracy on the cue-contradicting
questions (correct option not the tied-shortest; chance among the other four 25%), with bootstrap intervals clustered by component, for TF-IDF models and single style-feature groups.
python a33_wording.py -> analysis/a33_summary.json"""
import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
HEDGE = {"suggest", "suggests", "suggesting", "possible", "possibly", "likely", "probable", "probably", "apparent", "apparently", "may", "might", "could", "potential", "potentially", "consistent", "indicating", "indicates", "appears", "tenuous", "partial", "partially"}
NEG = re.compile(r"^(no|not|there is no|without|absent)\b", re.I)
TOK = re.compile(r"[a-z0-9]+")
rng = np.random.RandomState(20261103)
D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
COMP = json.load(open(ROOT + "/data/train_pool/split_v1/components.json"))
Q = []
for c in D:
    if c.get("question type") != "mcq":
        continue
    opts = [LAB.sub("", c[l]).strip() for l in "ABCDE"]
    L = [len(o) for o in opts]
    g = "ABCDE".index(c["answer"].strip())
    Q.append({"id": c["case_id"], "q": c["question"], "opts": opts, "gold": g, "short": min(range(5), key=lambda k: (L[k], k)), "contra": L[g] != min(L), "grp": COMP.get(c["case_id"], c["case_id"])})
print("questions", len(Q), "cue-contradicting", sum(x["contra"] for x in Q), flush=True)


def toks(s):
    return TOK.findall(s.lower())


def style(x):
    qt = set(toks(x["q"]))
    T = [toks(o) or ["_"] for o in x["opts"]]
    rows = []
    for k, o in enumerate(x["opts"]):
        t = T[k]
        others = set(w for j in range(5) if j != k for w in T[j])
        rows.append([sum(w in HEDGE for w in t) / len(t), o.count(",") / len(t), sum(w in qt for w in t) / len(t), sum(w in others for w in t) / len(t),
                     np.mean([len(set(t) & set(T[j])) / max(1, len(set(t) | set(T[j]))) for j in range(5) if j != k]), float(bool(NEG.match(o))), float(any(ch.isdigit() for ch in o))])
    M = np.array(rows, float)
    return M - M.mean(0)


S = np.vstack([style(x) for x in Q])  # (5N, 7)
GROUPS = {"hedging": [0], "commas": [1], "overlap_with_question": [2], "overlap_with_other_options": [3, 4], "negation_start": [5], "digits": [6], "all_style": list(range(7))}
Y = np.array([int(k == x["gold"]) for x in Q for k in range(5)])
OPT = [o for x in Q for o in x["opts"]]
grp = np.array([x["grp"] for x in Q])
contra = np.array([x["contra"] for x in Q])
KINDS = ["tfidf_all", "tfidf_contra"] + ["style_" + g + "_contra" for g in GROUPS]
oof = {k: np.zeros((len(Q), 5)) for k in KINDS}
kf = GroupKFold(5)
for tri, tei in kf.split(np.arange(len(Q)), groups=grp):
    rows_tr = (tri[:, None] * 5 + np.arange(5)).ravel()
    rows_te = (tei[:, None] * 5 + np.arange(5)).ravel()
    tri_c = tri[contra[tri]]
    rows_trc = (tri_c[:, None] * 5 + np.arange(5)).ravel()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xa = vec.fit_transform([OPT[i] for i in rows_tr])
    Xte = vec.transform([OPT[i] for i in rows_te])
    oof["tfidf_all"][tei] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xa, Y[rows_tr]).decision_function(Xte).reshape(-1, 5)
    vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xc = vec2.fit_transform([OPT[i] for i in rows_trc])
    oof["tfidf_contra"][tei] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xc, Y[rows_trc]).decision_function(vec2.transform([OPT[i] for i in rows_te])).reshape(-1, 5)
    for gname, cols in GROUPS.items():
        A, B = S[rows_trc][:, cols], S[rows_te][:, cols]
        mu, sd = A.mean(0), A.std(0) + 1e-9
        oof["style_" + gname + "_contra"][tei] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((A - mu) / sd, Y[rows_trc]).decision_function((B - mu) / sd).reshape(-1, 5)
idx = np.where(contra)[0]
ug = np.unique(grp[idx])
out = {"n_questions": len(Q), "n_contra": int(len(idx)), "n_components_contra": int(len(ug)), "chance_pct": 25.0, "models": {}}
gold = np.array([x["gold"] for x in Q])
short = np.array([x["short"] for x in Q])
byg = {g: idx[grp[idx] == g] for g in ug}
for k in KINDS:
    pred = oof[k].argmax(1)
    ok = (pred[idx] == gold[idx]).astype(float)
    ps = (pred[idx] == short[idx]).astype(float)
    boots = []
    for _ in range(1000):
        pick = rng.choice(len(ug), len(ug))
        sel = np.concatenate([byg[ug[j]] for j in pick])
        boots.append(100 * float((pred[sel] == gold[sel]).mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["models"][k] = {"acc_contra": round(100 * float(ok.mean()), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "picks_shortest": round(100 * float(ps.mean()), 2)}
    print(k, json.dumps(out["models"][k]), flush=True)
# descriptive: strongest word / bigram coefficients on the cue-contradicting questions (all data)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
rows_c = (idx[:, None] * 5 + np.arange(5)).ravel()
X = vec.fit_transform([OPT[i] for i in rows_c])
clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, Y[rows_c])
names = np.array(vec.get_feature_names_out())
o = np.argsort(clf.coef_[0])
out["top_terms_correct"] = [str(t) for t in names[o[::-1][:25]]]
out["top_terms_wrong"] = [str(t) for t in names[o[:25]]]
m = out["models"]
out["V1_replicates_at_scale"] = bool(m["tfidf_all"]["acc_contra"] >= 40.0 and m["tfidf_all"]["ci95"][0] >= 35.0 and m["tfidf_contra"]["acc_contra"] >= 35.0 and m["tfidf_contra"]["ci95"][0] >= 30.0)
out["V2_single_groups_with_lower_bound_gt_25"] = [g for g in GROUPS if m["style_" + g + "_contra"]["ci95"][0] > 25.0]
json.dump(out, open(ROOT + "/analysis/a33_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
