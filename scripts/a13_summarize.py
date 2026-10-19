#!/usr/bin/env python
"""A13 analysis (planted length cue on PMC-VQA). python a13_summarize.py RUN_DIR MODEL_KEY(q3b|q7b|ivl2b)"""
import glob
import json
import math
import os
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_implant import implant_options  # noqa: E402

LET = "ABCD"
run_dir, key = sys.argv[1], sys.argv[2]
REC = json.load(open(ROOT + "/data/pmcvqa/split_a11/records.json"))
R = {}
for g in REC.values():
    R.update(g)


def read_jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def softmax(l):
    m = max(l)
    e = [math.exp(x - m) for x in l]
    s = sum(e)
    return [x / s for x in e]


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def pred(l):
    return LET[max(range(4), key=lambda k: l[k])]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n))


def table(rows, group):
    d = {}
    for r in rows:
        if r["group"] == group:
            d.setdefault(r["arm"], {})[r["case_id"]] = r["direct"]["lp"]
    return d


def shortest(cid, cond):
    ch = implant_options(cid, R[cid]["choices"], LET.index(R[cid]["gold"]))[0] if cond == "implant" else R[cid]["choices"]
    L = [len(c) for c in ch]
    m = min(L)
    return LET[L.index(m)] if L.count(m) == 1 else None


def eff(t, ids):
    gold = {i: R[i]["gold"] for i in ids}
    ct = {i: pred(t["base"][i]) == gold[i] for i in ids}
    cb = {i: pred(t["black"][i]) == gold[i] for i in ids}
    fx = sum(ct[i] and not cb[i] for i in ids)
    br = sum((not ct[i]) and cb[i] for i in ids)
    m = len(ids)
    return {"n": m, "acc_true": round(sum(ct.values()) / m, 4), "acc_black": round(sum(cb.values()) / m, 4), "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2),
            "fixes": fx, "breaks": br, "mcnemar_p": mcnemar_exact(fx, br) if fx + br else None}


imp = read_jsonl(os.path.join(run_dir, "rows_ft_implant.jsonl"))
org = read_jsonl(os.path.join(run_dir, "rows_ft_orig.jsonl"))
tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
a11 = json.load(open(glob.glob(ROOT + f"/experiments/pilot/pmc_{key}_s0_22584*/summary.json")[0]))["ft"]["E1"]
out = {"run_dir": run_dir, "model": key, "training": {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]},
       "a11_reference_E1": {"acc_true": a11["acc"]["base"], "gap_points": a11["all_black"]["diff_points"]}}
val = table(imp, "VAL700")["base"]
ms = sorted(margin(l) for l in val.values())
tau = ms[int(0.15 * len(ms))]
out["tau_m_model"] = tau
for name, rows, cond in (("E1_implant", imp, "implant"), ("E1_orig", org, "orig"), ("E2_implant", imp, "implant")):
    g = name.split("_")[0]
    t = table(rows, g)
    ids = sorted(set(t["base"]) & set(t["black"]))
    mb = {i: margin(t["black"][i]) for i in ids}
    d = eff(t, ids)
    d["share_black_margin_ge_0.99"] = round(sum(mb[i] >= 0.99 for i in ids) / len(ids), 4)
    d["share_black_margin_ge_0.50"] = round(sum(mb[i] >= 0.50 for i in ids) / len(ids), 4)
    sh = {i: shortest(i, cond) for i in ids}
    d["pred_equals_shortest"] = round(sum(pred(t["base"][i]) == sh[i] for i in ids) / len(ids), 4)
    d["gold_is_shortest"] = round(sum(R[i]["gold"] == sh[i] for i in ids) / len(ids), 4)
    strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}
    d["strata_sizes"] = {s: sum(v == s for v in strat.values()) for s in ("HI", "MID", "LO")}
    lo = [i for i in ids if strat[i] == "LO"]
    d["LO"] = eff(t, lo) if lo else None
    hi = [i for i in ids if strat[i] == "HI"]
    d["HI_changed_share"] = round(sum(pred(t["base"][i]) != pred(t["black"][i]) for i in hi) / max(1, len(hi)), 4)
    out[name] = d
e = out["E1_implant"]
v = {"I1_habit": e["pred_equals_shortest"] >= 0.80,
     "I2_text_decisive": e["share_black_margin_ge_0.99"] >= 0.30,
     "I3_gap_halved": e["diff_points"] <= 0.5 * out["a11_reference_E1"]["gap_points"],
     "I4_acc_drop_on_original": (out["a11_reference_E1"]["acc_true"] - out["E1_orig"]["acc_true"]) >= 0.05}
out["verdicts"] = v
json.dump(out, open(os.path.join(run_dir, "summary_a13.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:5000])
