#!/usr/bin/env python
"""A10 analysis for one recipe variant. python ft_summarize2.py RUN_DIR"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, softmax, mcnemar_exact, gold_from_train, shortest_letter, LETTERS

run_dir = sys.argv[1]
T = gold_from_train()
cfg = json.load(open(os.path.join(run_dir, "config.json")))
rows = read_jsonl(os.path.join(run_dir, "rows_ft.jsonl"))
tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def pred(l):
    return LETTERS[max(range(5), key=lambda k: l[k])]


def table(group):
    d = {}
    for r in rows:
        if r["group"] == group:
            d.setdefault(r["arm"], {})[r["case_id"]] = r["direct"]["lp"]
    return d


def cp_upper(k, n, a=0.05):
    if k >= n:
        return 1.0
    cdf = lambda x, p: sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, x + 1))
    lo, hi = k / n, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if cdf(k, m) < a / 2 else (m, hi)
    return round((lo + hi) / 2, 4)


out = {"run_dir": run_dir, "variant": cfg.get("variant"), "model": cfg["model"], "recipe": cfg["recipe"], "sizes": cfg["sizes"],
       "training": {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]}}
val = table("val700")["base"]
ms = sorted(margin(l) for l in val.values())
tau = ms[int(0.15 * len(ms))]
out["tau_m_model"] = tau
t = table("test711")
ids = sorted(set(t["base"]) & set(t["black"]))
gold = {i: T[i]["answer"].strip() for i in ids}
short = {i: shortest_letter(T[i]) for i in ids}
n = len(ids)
out["TEST711"] = {"acc_base": round(sum(pred(t["base"][i]) == gold[i] for i in ids) / n, 4), "acc_black": round(sum(pred(t["black"][i]) == gold[i] for i in ids) / n, 4),
                  "share_pred_equals_shortest": round(sum(pred(t["base"][i]) == short[i] for i in ids) / n, 4)}
f = table("fresh")
ids = sorted(set(f["base"]) & set(f["black"]))
n = len(ids)
gold = {i: T[i]["answer"].strip() for i in ids}
mb = {i: margin(f["black"][i]) for i in ids}
mt = {i: margin(f["base"][i]) for i in ids}
strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}
S = {s: [i for i in ids if strat[i] == s] for s in ("HI", "MID", "LO")}


def eff(sub):
    ct = {i: pred(f["base"][i]) == gold[i] for i in sub}
    cb = {i: pred(f["black"][i]) == gold[i] for i in sub}
    fx = sum(ct[i] and not cb[i] for i in sub)
    br = sum((not ct[i]) and cb[i] for i in sub)
    m = len(sub)
    chg = sum(pred(f["base"][i]) != pred(f["black"][i]) for i in sub)
    return {"n": m, "acc_base": round(sum(ct.values()) / m, 4), "acc_black": round(sum(cb.values()) / m, 4),
            "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2), "fixes": fx, "breaks": br,
            "mcnemar_p": round(mcnemar_exact(fx, br), 6) if (fx + br) else None, "changed": chg, "changed_share": round(chg / m, 4), "changed_upper95": cp_upper(chg, m)}


out["fresh"] = {"n": n, "acc_base": round(sum(pred(f["base"][i]) == gold[i] for i in ids) / n, 4), "strata_sizes": {s: len(v) for s, v in S.items()},
                "all_black": eff(ids), "by_stratum_black": {s: eff(v) for s, v in S.items() if v}}
errs = [i for i in ids if pred(f["base"][i]) != gold[i]]
flag = [i for i in ids if mt[i] < tau]
out["fresh"]["error_alarm"] = {"errors": len(errs), "flagged_fraction": round(len(flag) / n, 4), "captured": sum(i in set(flag) for i in errs),
                               "capture_fraction": round(sum(i in set(flag) for i in errs) / max(1, len(errs)), 4)}
lo, hi = out["fresh"]["by_stratum_black"].get("LO"), out["fresh"]["by_stratum_black"].get("HI")
v = {}
v["R1_falsified_for_this_variant"] = bool(lo and lo["diff_points"] >= 5.0 and lo["mcnemar_p"] is not None and lo["mcnemar_p"] < 0.05)
v["R2_habit"] = out["TEST711"]["share_pred_equals_shortest"] >= 0.80
v["R3_saturation"] = bool(hi and hi["changed_share"] <= 0.01)
ea = out["fresh"]["error_alarm"]
v["R4_alarm"] = bool(0.10 <= ea["flagged_fraction"] <= 0.25 and ea["capture_fraction"] >= 0.80)
out["verdicts"] = v
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
