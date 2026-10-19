#!/usr/bin/env python
"""A36 part 2: blind ceiling and visual headroom of seven multiple-choice benchmarks. Blind ceiling B_opt = accuracy of an option-level classifier (word and bigram TF-IDF of the option plus four within-question length features;
no question, no image) on the benchmark's test set; visual headroom = accuracy of the fine-tuned Qwen-3B with the true image minus max(B_opt, its accuracy with a black image).
python a36_blind.py -> analysis/a36_blind_summary.json"""
import ast
import csv
import glob
import json
import re
import zipfile

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
D = ROOT + "/data"
LAB = re.compile(r"^\s*\(?[A-Za-z]\)?[\.\):]\s*")
rng = np.random.RandomState(20261105)


def strip_letter(s):
    return LAB.sub("", str(s)).strip()


def rows(f):
    return pq.read_table(f).to_pylist()


def load_medcmr():
    ids = set(json.load(open(D + "/train_pool/split_v1/test_ids.json")))
    tr, te = [], []
    for c in json.load(open(D + "/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]:
        if c.get("question type") != "mcq":
            continue
        opts = [strip_letter(c[l]) for l in "ABCDE"]
        (te if c["case_id"] in ids else tr).append((opts, "ABCDE".index(c["answer"].strip())))
    return tr, te


def load_pmc():
    def rd(f):
        out = []
        for r in csv.DictReader(open(f"{D}/pmcvqa/{f}", encoding="utf-8", newline="")):
            opts = [strip_letter(r[f"Choice {x}"]) for x in "ABCD" if r.get(f"Choice {x}")]
            lab = str(r["Answer_label"]).strip()
            if len(opts) == 4 and lab and lab in "ABCD":
                out.append((opts, "ABCD".index(lab)))
        return out
    tr = rd("train.csv")
    idx = rng.permutation(len(tr))[:30000]
    return [tr[i] for i in idx], rd("test_clean.csv")


def load_omni():
    z = zipfile.ZipFile(D + "/omnimed/OmniMedVQA.zip")
    items = []
    for n in z.namelist():
        if "QA_information/Open-access/" in n and n.endswith(".json"):
            for q in json.loads(z.read(n)):
                opts = [str(q.get(f"option_{x}")) for x in "ABCD" if q.get(f"option_{x}") is not None]
                if q.get("gt_answer") in opts:
                    items.append((opts, opts.index(q["gt_answer"])))
    p = rng.permutation(len(items))
    return [items[i] for i in p[2000:32000]], [items[i] for i in p[:2000]]


def load_aok():
    f = lambda pat: [(list(r["choices"]), int(r["correct_choice_idx"])) for x in glob.glob(f"{D}/aokvqa/raw/data/{pat}") for r in rows(x)]
    return f("train-*.parquet"), f("validation-*.parquet")


def load_medx():
    items = []
    for ln in open(f"{D}/cue_audit/medxpert_test.jsonl"):
        q = json.loads(ln)
        ks = sorted(q["options"])
        items.append(([q["options"][k] for k in ks], ks.index(q["label"])))
    return None, items


def load_mmmu():
    items = []
    for f in glob.glob(f"{D}/cue_audit/mmmu_*.parquet"):
        for r in rows(f):
            if r["question_type"] == "multiple-choice":
                o = ast.literal_eval(r["options"])
                items.append((o, "ABCDEFGHI".index(r["answer"])))
    return None, items


def lenfeats(opts):
    L = np.array([len(o) for o in opts], float)
    W = np.array([len(o.split()) for o in opts], float)
    return np.stack([L - L.mean(), W - W.mean(), np.argsort(np.argsort(L)) - (len(opts) - 1) / 2.0, (L - L.mean()) / (L.std() + 1e-6)], 1)


def fit_eval(tr, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3 if len(tr) > 1000 else 2, sublinear_tf=True)
    Xt = vec.fit_transform([o for opts, _ in tr for o in opts])
    Lt = np.vstack([lenfeats(o) for o, _ in tr])
    mu, sd = Lt.mean(0), Lt.std(0) + 1e-9
    y = np.array([int(k == g) for opts, g in tr for k in range(len(opts))])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(hstack([Xt, csr_matrix((Lt - mu) / sd)]).tocsr(), y)
    Xe = hstack([vec.transform([o for opts, _ in te for o in opts]), csr_matrix((np.vstack([lenfeats(o) for o, _ in te]) - mu) / sd)]).tocsr()
    s = clf.decision_function(Xe)
    ok, p = [], 0
    for opts, g in te:
        ok.append(int(np.argmax(s[p:p + len(opts)]) == g))
        p += len(opts)
    return np.array(ok, float)


def cv_eval(items):
    ok = np.zeros(len(items))
    for a, b in KFold(5, shuffle=True, random_state=20261105).split(items):
        ok[b] = fit_eval([items[i] for i in a], [items[i] for i in b])
    return ok


BENCH = [("Med-CMR", load_medcmr), ("PMC-VQA", load_pmc), ("OmniMedVQA", load_omni), ("A-OKVQA", load_aok), ("MedXpertQA-MM", load_medx), ("MMMU health", load_mmmu)]
out = {"benchmarks": {}}
for name, loader in BENCH:
    tr, te = loader()
    ok = cv_eval(te) if tr is None else fit_eval(tr, te)
    chance = 100 * float(np.mean([1.0 / len(o) for o, _ in te]))
    boots = [100 * ok[rng.randint(0, len(ok), len(ok))].mean() for _ in range(1000)]
    out["benchmarks"][name] = {"n_test": len(te), "n_train": 0 if tr is None else len(tr), "protocol": "5-fold CV within the test set" if tr is None else "train/test", "chance": round(chance, 1),
                               "B_opt": round(100 * float(ok.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}
    print(name, json.dumps(out["benchmarks"][name]), flush=True)


def accs(pat, group, arms, key="rows_ft.jsonl", lo=False):
    d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}") if "smoke" not in x and glob.glob(x + "/" + key)][0]
    P = {}
    for ln in open(d + "/" + key):
        r = json.loads(ln)
        if r["group"].lower() == group.lower() and r["arm"] in arms:
            lp = r["direct"]["lp"]
            P.setdefault(r["arm"], []).append(int(int(np.argmax(lp)) == "ABCDE".index(r["gold"])))
    return {a: round(100 * float(np.mean(v)), 2) for a, v in P.items()}


FT = {"Med-CMR": accs("ft_3b_s0_22569048_*", "fresh", ("base", "black")), "PMC-VQA": accs("pmc_q3b_s0_*", "E1", ("base", "black")), "OmniMedVQA": accs("omni16_q3b_seed20261003_*", "E1", ("base", "black"))}
d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a21_q3b_length0.25_seed20261021_2*/summary_dose.json")][0]
o = json.load(open(d))["orig_E1"]
FT["A-OKVQA"] = {"base": o["acc"], "black": o["acc_black"]}
for k, v in FT.items():
    b = out["benchmarks"][k]
    b["ft_true"], b["ft_black"] = v["base"], v["black"]
    b["visual_headroom"] = round(v["base"] - max(b["B_opt"], v["black"]), 2)
B = out["benchmarks"]
out["H1_blind_ceilings"] = bool(B["Med-CMR"]["B_opt"] >= 85.0 and B["PMC-VQA"]["B_opt"] <= 45.0 and B["A-OKVQA"]["B_opt"] <= 45.0)
hr = {k: B[k]["visual_headroom"] for k in FT}
out["H2_medcmr_smallest_headroom"] = bool(min(hr, key=hr.get) == "Med-CMR")
json.dump(out, open(ROOT + "/analysis/a36_blind_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "benchmarks"}))
