#!/usr/bin/env python
"""Acceptance checks on the official MedReason package + preregistered H3. CPU only, no model.
H3 (docs/preregistration_v1.md + amendment A0): shortest-option rule accuracy on Pool MCQs lies in [0.88, 0.94].
Rule = phase2c: min over options of (character length, letter); label prefix stripped (it has equal length for all options).
Writes analysis/acceptance_and_H3.json and .md"""
import collections, csv, hashlib, json, math, os, re, statistics as st

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
LAB = re.compile(r"^\s*[A-E]\)\s*")
res = {"checks": {}}


def chk(name, ok, detail=""):
    res["checks"][name] = {"pass": bool(ok), "detail": str(detail)}
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


T = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
V = json.load(open(POOL + "/official_json/medreason_validation_participant_facing.json", encoding="utf-8"))["cases"]
H = [l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()]
hs = set(H)
tid, vid = [c["case_id"] for c in T], [c["case_id"] for c in V]
byid = {c["case_id"]: c for c in T}

chk("train_count_17722", len(T) == 17722, len(T))
chk("val_count_2532", len(V) == 2532, len(V))
cq = collections.Counter(c["question type"] for c in T)
chk("train_mcq_open_14398_3324", cq["mcq"] == 14398 and cq["open-ended"] == 3324, dict(cq))
cv = collections.Counter(c["question type"] for c in V)
chk("val_mcq_open_2057_475", cv["mcq"] == 2057 and cv["open-ended"] == 475, dict(cv))
chk("ids_unique", len(set(tid)) == len(tid) and len(set(vid)) == len(vid))
chk("train_val_disjoint", not (set(tid) & set(vid)))
chk("holdout_886_subset_of_train", hs <= set(tid), len(hs & set(tid)))
chk("holdout_disjoint_from_val", not (hs & set(vid)))
pool_ids = [i for i in tid if i not in hs]
chk("authorised_pool_is_16836", len(pool_ids) == 16836, len(pool_ids))

# holdout records agree with official train records
cases = json.load(open(ROOT + "/results/reproduction_frozen/data_identity/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
feat = {r["case_id"]: r for r in csv.DictReader(open(ROOT + "/analysis/mcq_all711_features.csv", encoding="utf-8"))}
qm = mm = om = am = nm = 0
for c in cases:
    r = byid[c["case_id"]]
    qm += (c["question"] == r["question"])
    md = (c.get("metadata") or {}).get("modality")
    mm += (md is None or str(md) == str(r["modality"]))
    if c["task_type"] == "mcq":
        nm += 1
        ot = {o["label"]: LAB.sub("", o["text"]).strip() for o in c["options"]}
        rt = {k: LAB.sub("", r[k]).strip() for k in "ABCDE"}
        om += (ot == rt)
        am += (feat[c["case_id"]]["ground_truth"].strip() == r["answer"].strip())
chk("holdout_question_equals_train_record", qm == 886, qm)
chk("holdout_modality_equals_train_record", mm == 886, mm)
chk("holdout_mcq_options_equal_train_record", om == nm == 711, "%d/%d" % (om, nm))
chk("holdout_mcq_answers_equal_train_record", am == nm, "%d/%d" % (am, nm))

# record validity
bad = 0
for c in T:
    if c["question type"] == "mcq":
        labs = [k for k in "ABCDE" if c.get(k)]
        bad += (len(labs) != 5 or c["answer"] not in labs)
    else:
        bad += (not str(c["answer"]).strip())
chk("train_records_valid_k5_and_nonempty_answers", bad == 0, "bad=%d" % bad)
chk("val_has_no_answers", not any("answer" in c for c in V))

# images present, equal to manifest, decodable
man = json.load(open(POOL + "/IMAGE_MANIFEST.json"))
miss = mism = undec = 0
try:
    from PIL import Image
except ImportError:
    Image = None
for split, cs in (("train", T), ("val", V)):
    for c in cs:
        name = os.path.basename(c["image_path"])
        p = POOL + "/extracted/" + split + "/imgs/" + name
        if not os.path.exists(p):
            miss += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        mism += (h.hexdigest() != man[split].get(name))
        if Image is not None:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                undec += 1
chk("images_present", miss == 0, "missing=%d" % miss)
chk("images_match_manifest_sha256", mism == 0, "mismatch=%d" % mism)
chk("images_decodable", Image is not None and undec == 0, "undecodable=%d PIL=%s" % (undec, Image is not None))
res["modality_counts_train"] = dict(collections.Counter(c["modality"] for c in T))
res["task_counts_train_pool"] = dict(collections.Counter(byid[i]["task"] for i in pool_ids))
res["fields_train"] = sorted(T[0].keys())
res["has_patient_or_study_id_field"] = any(k.lower() in ("patient_id", "study_id", "figure_id", "article_id", "source_id") for k in T[0])


# ------------------------------------------------------------------ H3
def opts(c):
    return {k: LAB.sub("", c[k]).strip() for k in "ABCDE"}


def rule_char(c):
    o = opts(c)
    return min(o, key=lambda k: (len(o[k]), k))


def acc(cs, fn=rule_char):
    k = sum(fn(c) == c["answer"].strip() for c in cs)
    return k, len(cs)


test_mcq = [byid[i] for i in H if i.startswith("MCQ_")]
pool_mcq = [byid[i] for i in pool_ids if byid[i]["question type"] == "mcq"]
k_t, n_t = acc(test_mcq)
chk("sanity_rule_reproduces_648_of_711_on_test", (k_t, n_t) == (648, 711), "%d/%d" % (k_t, n_t))
k, n = acc(pool_mcq)
a = k / n
ci = wilson(k, n)
h3 = 0.88 <= a <= 0.94
res["H3"] = {"definition": "char-length shortest option, ties to first letter (A0)", "pool_mcq_n": n, "correct": k, "accuracy": round(a, 4),
             "wilson95": ci, "predicted_interval": [0.88, 0.94], "supported": h3,
             "test_mcq_same_rule": {"correct": k_t, "n": n_t, "accuracy": round(k_t / n_t, 4)}}
print("H3: %d/%d = %.4f  CI %s  interval [0.88,0.94] -> %s" % (k, n, a, ci, "SUPPORTED" if h3 else "FALSIFIED"), flush=True)

# exploratory secondary statistics (pool, label-using but outside the test set)
ex = {}


def rank_chars(c):
    o = opts(c)
    g = len(o[c["answer"].strip()])
    return 1 + sum(len(v) < g for v in o.values())


ex["gold_char_length_rank_pool"] = dict(sorted(collections.Counter(rank_chars(c) for c in pool_mcq).items()))
ex["longest_option_accuracy_pool"] = {"correct": sum(max(opts(c), key=lambda k, o=opts(c): (len(o[k]), k)) == c["answer"].strip() for c in pool_mcq), "n": n}
uniq = tied = ok = 0
for c in pool_mcq:
    o = opts(c)
    w = {k: len(v.split()) for k, v in o.items()}
    m = min(w.values())
    idx = [k for k, v in w.items() if v == m]
    if len(idx) == 1:
        uniq += 1
        ok += (idx[0] == c["answer"].strip())
    else:
        tied += 1
ex["shortest_by_words_pool"] = {"unique_cases": uniq, "unique_is_gold": ok, "acc_on_unique": round(ok / uniq, 4), "tied_cases": tied}
ratios = []
for c in pool_mcq:
    o = opts(c)
    g = c["answer"].strip()
    d = [len(v) for k, v in o.items() if k != g]
    ratios.append(len(o[g]) / (sum(d) / len(d)))
ex["median_gold_over_mean_distractor_chars_pool"] = round(st.median(ratios), 3)
by_task = {}
for c in pool_mcq:
    by_task.setdefault(c["task"], []).append(c)
ex["shortest_rule_accuracy_by_task_pool"] = {t: {"n": len(v), "acc": round(acc(v)[0] / len(v), 4)} for t, v in sorted(by_task.items())}
by_mod = {}
for c in pool_mcq:
    by_mod.setdefault(c["primary_modality"], []).append(c)
ex["shortest_rule_accuracy_by_primary_modality_pool"] = {t: {"n": len(v), "acc": round(acc(v)[0] / len(v), 4)} for t, v in sorted(by_mod.items(), key=lambda x: -len(x[1])) if len(v) >= 100}
res["exploratory_secondary_pool"] = ex
res["all_checks_pass"] = all(v["pass"] for v in res["checks"].values())
json.dump(res, open(ROOT + "/analysis/acceptance_and_H3.json", "w"), indent=2)
print("ALL CHECKS PASS" if res["all_checks_pass"] else "SOME CHECKS FAILED", flush=True)
print(json.dumps({"H3": res["H3"], "exploratory": ex}, indent=1), flush=True)
