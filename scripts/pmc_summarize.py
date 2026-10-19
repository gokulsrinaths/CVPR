#!/usr/bin/env python
"""A11 analysis for one PMC-VQA run (zero-shot and fine-tuned stages). python pmc_summarize.py RUN_DIR"""
import json
import math
import os
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LET = "ABCD"
run_dir = sys.argv[1]
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


def cp_upper(k, n, a=0.05):
    if k >= n:
        return 1.0
    cdf = lambda x, p: sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log(1 - p)) for j in range(0, x + 1))
    lo, hi = k / n, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if cdf(k, m) < a / 2 else (m, hi)
    return round((lo + hi) / 2, 4)


def auroc(score_err, y):
    """AUROC of score for y=1 (errors); ties count half."""
    pos = [s for s, t in zip(score_err, y) if t]
    neg = [s for s, t in zip(score_err, y) if not t]
    if not pos or not neg:
        return None
    srt = sorted(range(len(score_err)), key=lambda i: score_err[i])
    ranks = [0.0] * len(srt)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and score_err[srt[j + 1]] == score_err[srt[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[srt[k]] = (i + j) / 2 + 1
        i = j + 1
    sp = sum(r for r, t in zip(ranks, y) if t)
    return round((sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def unique_shortest(cid):
    L = [len(c) for c in R[cid]["choices"]]
    m = min(L)
    return LET[L.index(m)] if L.count(m) == 1 else None


def table(rows, group):
    d = {}
    for r in rows:
        if r["group"] == group:
            d.setdefault(r["arm"], {})[r["case_id"]] = r["direct"]["lp"]
    return d


def analyse(rows, tau):
    res = {}
    for g in ("E1", "E2"):
        t = table(rows, g)
        ids = sorted(set(t["base"]) & set(t["black"]) & set(t["mismatch_rand"]) & set(t["mismatch_nn"]))
        n = len(ids)
        gold = {i: R[i]["gold"] for i in ids}
        P = {a: {i: pred(t[a][i]) for i in ids} for a in t}
        mb = {i: margin(t["black"][i]) for i in ids}
        mt = {i: margin(t["base"][i]) for i in ids}
        acc = {a: round(sum(P[a][i] == gold[i] for i in ids) / n, 4) for a in P}

        def eff(sub, arm="black"):
            ct = {i: P["base"][i] == gold[i] for i in sub}
            cb = {i: P[arm][i] == gold[i] for i in sub}
            fx = sum(ct[i] and not cb[i] for i in sub)
            br = sum((not ct[i]) and cb[i] for i in sub)
            m = len(sub)
            chg = sum(P["base"][i] != P[arm][i] for i in sub)
            return {"n": m, "acc_base": round(sum(ct.values()) / m, 4), "acc_alt": round(sum(cb.values()) / m, 4), "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2),
                    "fixes": fx, "breaks": br, "mcnemar_p": round(mcnemar_exact(fx, br), 6) if fx + br else None, "changed": chg, "changed_share": round(chg / m, 4), "changed_upper95": cp_upper(chg, m)} if m else {"n": 0}
        strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}
        S = {s: [i for i in ids if strat[i] == s] for s in ("HI", "MID", "LO")}
        band = [i for i in ids if mb[i] >= 0.99]
        fixed = {"lt0.5": [i for i in ids if mb[i] < 0.5], "0.5-0.9": [i for i in ids if 0.5 <= mb[i] < 0.9], "0.9-0.99": [i for i in ids if 0.9 <= mb[i] < 0.99], "ge0.99": band}
        errs = [i for i in ids if P["base"][i] != gold[i]]
        flag = {i for i in ids if mt[i] < tau}
        unf = [i for i in ids if i not in flag]
        er_f = sum(i in flag for i in errs) / max(1, len(flag))
        er_u = sum(i not in flag for i in errs) / max(1, len(unf))
        us = {i: unique_shortest(i) for i in ids}
        res[g] = {"n": n, "acc": acc, "tau_m": tau, "gold_unique_shortest_share": round(sum(us[i] == gold[i] for i in ids) / n, 4),
                  "pred_equals_unique_shortest": round(sum(P["base"][i] == us[i] for i in ids) / n, 4),
                  "strata_sizes": {s: len(v) for s, v in S.items()}, "all_black": eff(ids), "all_rand": eff(ids, "mismatch_rand"), "all_nn": eff(ids, "mismatch_nn"),
                  "by_stratum_black": {s: eff(v) for s, v in S.items() if v}, "by_stratum_rand": {s: eff(v, "mismatch_rand") for s, v in S.items() if v}, "by_stratum_nn": {s: eff(v, "mismatch_nn") for s, v in S.items() if v},
                  "band_ge_0.99_black": eff(band) if band else {"n": 0}, "band_ge_0.99_rand": eff(band, "mismatch_rand") if band else {"n": 0}, "band_ge_0.99_nn": eff(band, "mismatch_nn") if band else {"n": 0},
                  "fixed_strata_black": {k: eff(v) for k, v in fixed.items() if v},
                  "alarm": {"errors": len(errs), "flagged_fraction": round(len(flag) / n, 4), "error_rate_flagged": round(er_f, 4), "error_rate_unflagged": round(er_u, 4),
                            "ratio": round(er_f / er_u, 3) if er_u > 0 else None, "capture_fraction": round(sum(i in flag for i in errs) / max(1, len(errs)), 4),
                            "auroc_margin_for_errors": auroc([-mt[i] for i in ids], [int(i in set(errs)) for i in ids])}}
    return res


out = {"run_dir": run_dir, "config": json.load(open(os.path.join(run_dir, "config.json")))}
tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
out["training"] = {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]}
for stage in ("zeroshot", "ft"):
    rows = read_jsonl(os.path.join(run_dir, f"rows_{stage}.jsonl"))
    val = table(rows, "VAL700")["base"]
    ms = sorted(margin(l) for l in val.values())
    tau = ms[int(0.15 * len(ms))]
    out[stage] = analyse(rows, tau)
    out[stage]["tau_m_model"] = tau
v = {}
for g in ("E1", "E2"):
    f = out["ft"][g]
    lo = f["by_stratum_black"].get("LO", {"n": 0})
    band = f["band_ge_0.99_black"]
    v[g] = {"Q1_no_habit": f["pred_equals_unique_shortest"] <= 0.35,
            "Q2_overall_gap": bool(f["all_black"]["diff_points"] >= 5.0 and (f["all_black"]["mcnemar_p"] if f["all_black"]["mcnemar_p"] is not None else 1) < 0.05),
            "Q3_LO_effect": bool(lo.get("n", 0) and lo["diff_points"] >= 5.0 and (lo["mcnemar_p"] if lo["mcnemar_p"] is not None else 1) < 0.05),
            "Q4_noveto_band": ("not_evaluable_n<100" if band.get("n", 0) < 100 else bool(band["changed_share"] <= 0.02)),
            "Q5_alarm": bool(f["alarm"]["ratio"] and f["alarm"]["ratio"] >= 1.5 and (f["alarm"]["auroc_margin_for_errors"] if f["alarm"]["auroc_margin_for_errors"] is not None else 0) >= 0.65)}
out["verdicts_ft"] = v
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out["verdicts_ft"], indent=1))
for stage in ("zeroshot", "ft"):
    for g in ("E1", "E2"):
        f = out[stage][g]
        print(stage, g, "acc", f["acc"], "tau", round(out[stage]["tau_m_model"], 4), "strata", f["strata_sizes"], "gap", f["all_black"]["diff_points"], "LO", f["by_stratum_black"].get("LO", {}).get("diff_points"),
              "band", f["band_ge_0.99_black"].get("n"), f["band_ge_0.99_black"].get("changed"), "short", f["pred_equals_unique_shortest"])
