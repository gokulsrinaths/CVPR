#!/usr/bin/env python
"""A9 analysis for one fine-tuned model. python ft_summarize.py RUN_DIR"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, softmax, mcnemar_exact, gold_from_train, shortest_letter, LETTERS

run_dir = sys.argv[1]
T = gold_from_train()
zs = read_jsonl(os.path.join(run_dir, "rows_zeroshot.jsonl"))
ft = read_jsonl(os.path.join(run_dir, "rows_ft.jsonl"))


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def table(rows, group):
    d = {}
    for r in rows:
        if r["group"] == group:
            d.setdefault(r["arm"], {})[r["case_id"]] = r["direct"]["lp"]
    return d


def pred(l):
    return LETTERS[max(range(5), key=lambda k: l[k])]


def cp_upper(k, n, a=0.05):
    if k >= n:
        return 1.0

    def cdf(x, p):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, x + 1))
    lo, hi = k / n, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if cdf(k, m) < a / 2 else (m, hi)
    return round((lo + hi) / 2, 4)


out = {"run_dir": run_dir, "config": json.load(open(os.path.join(run_dir, "config.json")))["sizes"]}
tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
out["training"] = {"first_loss": tl[0]["mean_loss"] if tl else None, "last_loss": tl[-1]["mean_loss"] if tl else None, "minutes": tl[-1]["minutes"] if tl else None}

# --- TEST711: zero-shot vs fine-tuned
def test_stats(rows, label):
    t = table(rows, "test711")
    if "base" not in t:
        return None
    ids = sorted(set(t["base"]) & set(t.get("black", t["base"])))
    gold = {i: T[i]["answer"].strip() for i in ids}
    short = {i: shortest_letter(T[i]) for i in ids}
    n = len(ids)
    pb = {i: pred(t["base"][i]) for i in ids}
    d = {"n": n, "acc_base": round(sum(pb[i] == gold[i] for i in ids) / n, 4), "share_pred_equals_shortest": round(sum(pb[i] == short[i] for i in ids) / n, 4)}
    if "black" in t:
        pk = {i: pred(t["black"][i]) for i in ids}
        d["acc_black"] = round(sum(pk[i] == gold[i] for i in ids) / n, 4)
        d["headline_gap_points"] = round(100 * (d["acc_base"] - d["acc_black"]), 2)
    return d


out["TEST711_zeroshot"] = test_stats(zs, "zs")
out["TEST711_finetuned"] = test_stats(ft, "ft")

# --- tau_m from VAL700 (unlabeled, true image)
val = table(ft, "val700")["base"]
ms = sorted(margin(l) for l in val.values())
tau = ms[int(0.15 * len(ms))]
out["tau_m_model"] = tau
out["val700_n"] = len(ms)

# --- FRESH3000 structure
f = table(ft, "fresh")
ids = sorted(set.intersection(*[set(v) for v in f.values()]))
n = len(ids)
gold = {i: T[i]["answer"].strip() for i in ids}
mb = {i: margin(f["black"][i]) for i in ids}
mt = {i: margin(f["base"][i]) for i in ids}
strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}
S = {s: [i for i in ids if strat[i] == s] for s in ("HI", "MID", "LO")}


def eff(sub, a):
    ct = {i: pred(f["base"][i]) == gold[i] for i in sub}
    cb = {i: pred(f[a][i]) == gold[i] for i in sub}
    fx = sum(ct[i] and not cb[i] for i in sub)
    br = sum((not ct[i]) and cb[i] for i in sub)
    m = len(sub)
    chg = sum(pred(f["base"][i]) != pred(f[a][i]) for i in sub)
    return {"n": m, "acc_base": round(sum(ct.values()) / m, 4) if m else None, "acc_" + a: round(sum(cb.values()) / m, 4) if m else None,
            "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2) if m else None, "fixes": fx, "breaks": br,
            "mcnemar_p": round(mcnemar_exact(fx, br), 6) if (fx + br) else None, "changed": chg, "changed_share": round(chg / m, 4) if m else None,
            "changed_upper95": cp_upper(chg, m) if m else None}


out["fresh"] = {"n": n, "strata_sizes": {s: len(v) for s, v in S.items()}, "acc_base": round(sum(pred(f["base"][i]) == gold[i] for i in ids) / n, 4),
                "all_black": eff(ids, "black"), "by_stratum_black": {s: eff(v, "black") for s, v in S.items() if v},
                "by_stratum_mismatch_nn": {s: eff(v, "mismatch_nn") for s, v in S.items() if v},
                "by_stratum_mismatch_rand": {s: eff(v, "mismatch_rand") for s, v in S.items() if v}}
errs = [i for i in ids if pred(f["base"][i]) != gold[i]]
flag = [i for i in ids if mt[i] < tau]
out["fresh"]["error_alarm"] = {"errors": len(errs), "flagged_fraction": round(len(flag) / n, 4), "captured": sum(i in set(flag) for i in errs),
                               "capture_fraction": round(sum(i in set(flag) for i in errs) / max(1, len(errs)), 4)}
chg = [i for i in ids if pred(f["base"][i]) != pred(f["black"][i])]
out["fresh"]["changes_total"] = {"n": len(chg), "share_outside_HI": round(sum(strat[i] != "HI" for i in chg) / max(1, len(chg)), 4)}

# --- verdicts (A9)
v = {}
zsd, ftd = out["TEST711_zeroshot"], out["TEST711_finetuned"]
v["G1"] = "SUPPORTED" if (ftd["share_pred_equals_shortest"] >= 0.80 and zsd["share_pred_equals_shortest"] <= 0.50) else "FALSIFIED"
ev = out["fresh"]["acc_base"] >= 0.80
v["G2_evaluable"] = bool(ev)
hi = out["fresh"]["by_stratum_black"].get("HI")
lo = out["fresh"]["by_stratum_black"].get("LO")
hn = out["fresh"]["by_stratum_mismatch_nn"].get("HI")
if ev:
    c1 = bool(hi and hi["changed_share"] <= 0.01)
    c2 = ("NOT EVALUABLE (LO < 50)" if (not lo or lo["n"] < 50) else bool(lo["diff_points"] >= 5.0 and lo["mcnemar_p"] is not None and lo["mcnemar_p"] < 0.05))
    c3 = bool(out["fresh"]["changes_total"]["n"] and out["fresh"]["changes_total"]["share_outside_HI"] >= 0.90)
    c4 = bool(hn and hn["changed_share"] <= 0.02)
    v["G3"] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "supported": bool(c1 and c2 is True and c3 and c4)}
    ea = out["fresh"]["error_alarm"]
    v["G4"] = "SUPPORTED" if (0.10 <= ea["flagged_fraction"] <= 0.25 and ea["capture_fraction"] >= 0.80) else "FALSIFIED"
else:
    v["G3"] = v["G4"] = "NOT EVALUABLE (accuracy < 0.80)"
out["verdicts"] = v
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
