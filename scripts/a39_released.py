#!/usr/bin/env python
"""A39: the released MedReason adapter (and our fine-tuned systems) on the counter-cue and blind-fail questions of the 711 held-out MCQs. python a39_released.py -> analysis/a39_summary.json"""
import glob
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
rng = np.random.RandomState(20261109)
D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
test_ids = set(json.load(open(ROOT + "/data/train_pool/split_v1/test_ids.json")))
Q = {}
for c in D:
    if c.get("question type") != "mcq":
        continue
    opts = [LAB.sub("", c[l]).strip() for l in "ABCDE"]
    L = [len(o) for o in opts]
    g = "ABCDE".index(c["answer"].strip())
    Q[c["case_id"]] = {"opts": opts, "gold": g, "contra": L[g] != min(L), "test": c["case_id"] in test_ids}
tr = [v for v in Q.values() if not v["test"]]
te_ids = sorted(i for i, v in Q.items() if v["test"])


def lenfeats(opts):
    L = np.array([len(o) for o in opts], float)
    W = np.array([len(o.split()) for o in opts], float)
    return np.stack([L - L.mean(), W - W.mean(), np.argsort(np.argsort(L)) - 2.0, (L - L.mean()) / (L.std() + 1e-6)], 1)


vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xt = vec.fit_transform([o for v in tr for o in v["opts"]])
Lt = np.vstack([lenfeats(v["opts"]) for v in tr])
mu, sd = Lt.mean(0), Lt.std(0) + 1e-9
y = np.array([int(k == v["gold"]) for v in tr for k in range(5)])
clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(hstack([Xt, csr_matrix((Lt - mu) / sd)]).tocsr(), y)
Xe = hstack([vec.transform([o for i in te_ids for o in Q[i]["opts"]]), csr_matrix((np.vstack([lenfeats(Q[i]["opts"]) for i in te_ids]) - mu) / sd)]).tocsr()
s = clf.decision_function(Xe).reshape(-1, 5)
blind_pred = {i: int(s[k].argmax()) for k, i in enumerate(te_ids)}
BLIND_FAIL = {i for i in te_ids if blind_pred[i] != Q[i]["gold"]}
CONTRA = {i for i in te_ids if Q[i]["contra"]}
print("test", len(te_ids), "counter-cue", len(CONTRA), "blind-fail", len(BLIND_FAIL), "both", len(CONTRA & BLIND_FAIL), flush=True)

PRED = {}
r = json.load(open(ROOT + "/data/holdout886/run_output/results.json"))["answers"]
PRED["Released MedReason adapter (Qwen3.5-9B)"] = {p["case_id"]: "ABCDE".index(p["answer"].strip()) for p in r if p["task_type"] == "mcq" and p["answer"].strip() in list("ABCDE")}
SYS = [("A9 Qwen-3B", "ft_3b_s0_22569048_*"), ("A9 Qwen-7B", "ft_7b_s0_22569049_*")] + [(f"UL Qwen-3B seed {s % 100}", f"a26_medcmr_ul_3b_seed{s}_2*") for s in (20261041, 20261042, 20261043)] + [(f"UL Qwen-7B seed {s % 100}", f"a26_medcmr_ul_7b_seed{s}_2*") for s in (20261051, 20261052)]
for name, pat in SYS:
    f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}/rows_ft.jsonl") if "smoke" not in x]
    if not f:
        continue
    P = {}
    for ln in open(f[0]):
        rr = json.loads(ln)
        if rr["group"].lower() == "test711" and rr["arm"] == "base":
            P[rr["case_id"]] = int(max(range(5), key=lambda k: rr["direct"]["lp"][k]))
    PRED[name] = P


def acc(P, sel):
    ids = [i for i in sel if i in P]
    v = np.array([P[i] == Q[i]["gold"] for i in ids], float)
    boots = [100 * v[rng.randint(0, len(v), len(v))].mean() for _ in range(2000)]
    return {"n": len(ids), "acc": round(100 * float(v.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}


out = {"n_test": len(te_ids), "n_counter_cue": len(CONTRA), "n_blind_fail": len(BLIND_FAIL), "systems": {}}
follow = set(te_ids) - CONTRA
for name, P in PRED.items():
    out["systems"][name] = {"all": acc(P, te_ids), "counter_cue": acc(P, CONTRA), "cue_following": acc(P, follow), "blind_fail": acc(P, BLIND_FAIL)}
    print(name, json.dumps(out["systems"][name]), flush=True)
R = out["systems"]["Released MedReason adapter (Qwen3.5-9B)"]
out["C2_released_counter_cue_le_80_and_gap_ge_15"] = bool(R["counter_cue"]["acc"] <= 80.0 and R["cue_following"]["acc"] - R["counter_cue"]["acc"] >= 15.0)
json.dump(out, open(ROOT + "/analysis/a39_summary.json", "w"), indent=1)
print(json.dumps({"C2": out["C2_released_counter_cue_le_80_and_gap_ge_15"]}))
