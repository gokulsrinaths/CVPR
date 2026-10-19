#!/usr/bin/env python
"""Implements the decision rules of docs/gate_B_prespec.md. CPU only.
   python audit_summarize.py validity RUN_DIR | h1 RUN_DIR | selftest"""
import glob
import json
import math
import os
import re
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LETTERS = "ABCDE"
LAB = re.compile(r"^\s*[A-E]\)\s*")


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def softmax(lp):
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    z = sum(e)
    return [x / z for x in e]


def margin(lp):
    p = sorted(softmax(lp), reverse=True)
    return p[0] - p[1]


def argmax(lp):
    return max(range(5), key=lambda i: lp[i])


def median(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def ks_2samp(a, b):
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        x = min(a[i], b[j])
        while i < na and a[i] <= x:
            i += 1
        while j < nb and b[j] <= x:
            j += 1
        d = max(d, abs(i / na - j / nb))
    ne = na * nb / (na + nb)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def conf_frac(m):
    return sum(x >= 0.90 for x in m) / len(m)


def suspect(m_group, m_val):
    """Frozen rule: in-sample suspected iff KS p<0.01 AND median(group)>median(val) AND c(group)-c(val)>=0.05."""
    d, p = ks_2samp(m_group, m_val)
    cg, cv = conf_frac(m_group), conf_frac(m_val)
    mg, mv = median(m_group), median(m_val)
    return {"n_group": len(m_group), "n_val": len(m_val), "ks_D": round(d, 4), "ks_p": p, "median_m_group": round(mg, 4),
            "median_m_val": round(mv, 4), "c_group": round(cg, 4), "c_val": round(cv, 4), "c_diff": round(cg - cv, 4),
            "criteria": {"ks_p_lt_0.01": p < 0.01, "median_higher": mg > mv, "c_diff_ge_0.05": (cg - cv) >= 0.05},
            "in_sample_suspected": bool(p < 0.01 and mg > mv and (cg - cv) >= 0.05)}


def gold_from_train():
    T = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
    return {c["case_id"]: c for c in T}


def validity(run_dir):
    rows = read_jsonl(os.path.join(run_dir, "forward_rows.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    T = gold_from_train()
    out = {"run_dir": run_dir, "n_rows": {g: len(v) for g, v in by.items()}}
    mval = [margin(r["direct"]["lp"]) for r in by["val700"]]
    mpool = [margin(r["direct"]["lp"]) for r in by["pool700"]]
    ga = []
    for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
        ga = [r for r in read_jsonl(pth) if r["arm"] == "base"]
    mtest = [margin(r["direct"]["lp"]) for r in ga]
    out["POOL_vs_VAL"] = suspect(mpool, mval)
    out["TEST_vs_VAL"] = suspect(mtest, mval)
    k = sum(LETTERS[argmax(r["direct"]["lp"])] == r["gold"] for r in by["pool700"])
    n = len(by["pool700"])
    ci = wilson(k, n)
    kt = sum(LETTERS[argmax(r["direct"]["lp"])] == T[r["case_id"]]["answer"].strip() for r in ga)
    out["accuracy_direct_argmax"] = {"POOL700": {"correct": k, "n": n, "acc": round(k / n, 4), "wilson95": ci},
                                      "TEST711_gateA": {"correct": kt, "n": len(ga), "acc": round(kt / len(ga), 4)}}
    out["dev_protocol_accuracy_trigger_POOL"] = bool(k / n >= 0.99 or ci[0] > 0.9606)
    out["VAL700_share_of_mass_on_letters_median"] = round(median([r["direct"]["mass"] for r in by["val700"]]), 4)
    out["POOL700_share_of_mass_on_letters_median"] = round(median([r["direct"]["mass"] for r in by["pool700"]]), 4)
    pool_bad = out["POOL_vs_VAL"]["in_sample_suspected"] or out["dev_protocol_accuracy_trigger_POOL"]
    out["VERDICT"] = {"POOL": "IN-SAMPLE SUSPECTED" if pool_bad else "no evidence of memorisation at this resolution",
                      "TEST": "IN-SAMPLE SUSPECTED" if out["TEST_vs_VAL"]["in_sample_suspected"] else "no evidence of memorisation at this resolution"}
    return out


def shortest_letter(c):
    o = {k: LAB.sub("", c[k]).strip() for k in LETTERS}
    return min(o, key=lambda k: (len(o[k]), k))


def h1(run_dir):
    rows = read_jsonl(os.path.join(run_dir, "forward_rows.jsonl"))
    T = gold_from_train()
    arm = {"base": {}, "black": {}}
    for r in rows:
        arm[r["arm"]][r["case_id"]] = r
    ids = sorted(set(arm["base"]) & set(arm["black"]))
    out = {"run_dir": run_dir, "n_cases": len(ids)}
    short = {i: shortest_letter(T[i]) for i in ids}
    gold = {i: T[i]["answer"].strip() for i in ids}
    wrong_short = [i for i in ids if short[i] != gold[i]]
    out["n_cases_shortest_wrong"] = len(wrong_short)
    for ro in ("direct", "cue", "empty"):
        def pred(a, i):
            return LETTERS[argmax(arm[a][i][ro]["lp"])]
        tb = sum(pred("base", i) == gold[i] for i in ids)
        tk = sum(pred("black", i) == gold[i] for i in ids)
        n = len(ids)
        b = sum(pred("base", i) == gold[i] and pred("black", i) != gold[i] for i in ids)
        c = sum(pred("base", i) != gold[i] and pred("black", i) == gold[i] for i in ids)
        out[ro] = {"acc_true": round(tb / n, 4), "wilson_true": wilson(tb, n), "acc_black": round(tk / n, 4), "wilson_black": wilson(tk, n),
                   "loss_points": round(100 * (tb - tk) / n, 2), "mcnemar_true_only": b, "mcnemar_black_only": c,
                   "mcnemar_p": mcnemar_exact(b, c),
                   "share_pred_equals_shortest_true": round(sum(pred("base", i) == short[i] for i in ids) / n, 4),
                   "share_pred_equals_shortest_black": round(sum(pred("black", i) == short[i] for i in ids) / n, 4),
                   "acc_true_on_shortest_wrong": round(sum(pred("base", i) == gold[i] for i in wrong_short) / max(1, len(wrong_short)), 4),
                   "acc_black_on_shortest_wrong": round(sum(pred("black", i) == gold[i] for i in wrong_short) / max(1, len(wrong_short)), 4)}
    p = out["direct"]
    if p["acc_true"] < 0.60:
        out["H1"] = "NOT EVALUABLE (acc_true < 0.60 on the primary readout)"
    else:
        out["H1"] = "SUPPORTED" if p["loss_points"] <= 5.0 else "FALSIFIED"
    return out


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n)
        if not ok:
            fails.append(n)
    import random
    r = random.Random(1)
    same_a = [min(1, abs(r.gauss(0.9, 0.15))) for _ in range(700)]
    same_b = [min(1, abs(r.gauss(0.9, 0.15))) for _ in range(700)]
    shifted = [min(1, abs(r.gauss(0.97, 0.05))) for _ in range(700)]
    check("same distribution -> not suspected", not suspect(same_a, same_b)["in_sample_suspected"])
    check("more confident group -> suspected", suspect(shifted, same_b)["in_sample_suspected"])
    check("less confident group -> not suspected", not suspect(same_b, shifted)["in_sample_suspected"])
    check("wilson(50,100) brackets 0.5", wilson(50, 100)[0] < 0.5 < wilson(50, 100)[1])
    check("mcnemar(0,0)=1 and mcnemar(8,1)<0.05", mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(8, 1) < 0.05)
    check("margin of [0,0,0,0,-50]-like dist", abs(margin([0, 0, 0, 0, -50]) - 0.0) < 1e-9)
    check("margin of confident dist", margin([10, 0, 0, 0, 0]) > 0.99)
    check("ks identical -> D=0", ks_2samp([1, 2, 3], [1, 2, 3])[0] == 0.0)
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    fn = {"validity": validity, "h1": h1}[sys.argv[1]]
    res = fn(sys.argv[2])
    json.dump(res, open(os.path.join(sys.argv[2], "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
