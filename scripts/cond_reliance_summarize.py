#!/usr/bin/env python
"""A8 analysis. python cond_reliance_summarize.py RUN_DIR"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, softmax, mcnemar_exact, gold_from_train, shortest_letter, LETTERS

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
run_dir = sys.argv[1]
TAU = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateD_retrieval_s0_22468934_*/config.json")[0]))["tau_m"]
T = gold_from_train()
rows = [r for r in read_jsonl(os.path.join(run_dir, "rows.jsonl")) if r["group"] == "fresh"]
arm = {a: {} for a in ("base", "black", "mismatch_rand", "mismatch_nn")}
for r in rows:
    arm[r["arm"]][r["case_id"]] = r
ids = sorted(set.intersection(*[set(v) for v in arm.values()]))
n = len(ids)


def lp(a, i, ro="direct"):
    return arm[a][i][ro]["lp"]


def pred(a, i, ro="direct"):
    l = lp(a, i, ro)
    return LETTERS[max(range(5), key=lambda k: l[k])]


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def cp_upper(k, nn_, a=0.05):
    if k >= nn_:
        return 1.0

    def cdf(x, p):
        return sum(math.comb(nn_, j) * p ** j * (1 - p) ** (nn_ - j) for j in range(0, x + 1))
    lo, hi = k / nn_, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if cdf(k, m) < a / 2 else (m, hi)
    return round((lo + hi) / 2, 4)


gold = {i: T[i]["answer"].strip() for i in ids}
mb = {i: margin(lp("black", i)) for i in ids}
mt = {i: margin(lp("base", i)) for i in ids}
strat = {i: ("HI" if mb[i] >= TAU else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}


def eff(sub, a="black"):
    ct = {i: pred("base", i) == gold[i] for i in sub}
    cb = {i: pred(a, i) == gold[i] for i in sub}
    f = sum(ct[i] and not cb[i] for i in sub)
    b = sum((not ct[i]) and cb[i] for i in sub)
    m = len(sub)
    chg = sum(pred("base", i) != pred(a, i) for i in sub)
    return {"n": m, "acc_base": round(sum(ct.values()) / m, 4) if m else None, "acc_" + a: round(sum(cb.values()) / m, 4) if m else None,
            "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2) if m else None, "image_fixes": f, "image_breaks": b,
            "mcnemar_p": round(mcnemar_exact(f, b), 5) if (f + b) else None, "changed": chg, "changed_share": round(chg / m, 4) if m else None,
            "changed_upper95": cp_upper(chg, m) if m else None}


def infl(sub):
    v = []
    for i in sub:
        lt, lb = lp("base", i), lp("black", i)
        top = sorted(range(5), key=lambda k: -lt[k])[:2]
        v.append(abs((lt[top[0]] - lt[top[1]]) - (lb[top[0]] - lb[top[1]])))
    v.sort()
    return {"median": round(v[len(v) // 2], 3), "p90": round(v[int(0.9 * len(v))], 3)} if v else None


S = {s: [i for i in ids if strat[i] == s] for s in ("HI", "MID", "LO")}
out = {"run_dir": run_dir, "n": n, "tau_m": TAU, "label": "FRESH3000 pool sample: IN-SAMPLE SUSPECTED for the adapter", "strata_sizes": {s: len(v) for s, v in S.items()}}
out["all_black"] = eff(ids, "black")
out["by_stratum_black"] = {s: eff(v, "black") for s, v in S.items() if v}
out["by_stratum_mismatch_nn"] = {s: eff(v, "mismatch_nn") for s, v in S.items() if v}
out["by_stratum_mismatch_rand"] = {s: eff(v, "mismatch_rand") for s, v in S.items() if v}
out["image_influence_log_odds_by_stratum"] = {s: infl(v) for s, v in S.items() if v}
errs = [i for i in ids if pred("base", i) != gold[i]]
flag = [i for i in ids if mt[i] < TAU]
out["error_alarm"] = {"errors": len(errs), "flagged": len(flag), "flagged_fraction": round(len(flag) / n, 4), "errors_captured": sum(i in set(flag) for i in errs),
                      "capture_fraction": round(sum(i in set(flag) for i in errs) / max(1, len(errs)), 4)}
ws = [i for i in ids if shortest_letter(T[i]) != gold[i]]
out["shortest_wrong_subset_black"] = eff(ws, "black")
changed_all = [i for i in ids if pred("base", i) != pred("black", i)]
out["changes_total"] = {"n_changed": len(changed_all), "in_mid_or_lo": sum(strat[i] != "HI" for i in changed_all),
                        "share_in_mid_or_lo": round(sum(strat[i] != "HI" for i in changed_all) / max(1, len(changed_all)), 4)}
v = {}
hi = out["by_stratum_black"].get("HI")
v["C1"] = "SUPPORTED" if hi and hi["changed_share"] <= 0.01 else "FALSIFIED"
lo = out["by_stratum_black"].get("LO")
v["C2"] = "SUPPORTED" if lo and lo["diff_points"] >= 5.0 and lo["mcnemar_p"] is not None and lo["mcnemar_p"] < 0.05 else "FALSIFIED"
v["C3"] = "SUPPORTED" if out["changes_total"]["n_changed"] and out["changes_total"]["share_in_mid_or_lo"] >= 0.90 else "FALSIFIED"
hn = out["by_stratum_mismatch_nn"].get("HI")
v["C4"] = "SUPPORTED" if hn and hn["changed_share"] <= 0.02 else "FALSIFIED"
v["C5"] = "SUPPORTED" if (out["all_black"]["diff_points"] < 3.0 and v["C2"] == "SUPPORTED") else "FALSIFIED"
ea = out["error_alarm"]
v["C6"] = "SUPPORTED" if (0.10 <= ea["flagged_fraction"] <= 0.25 and ea["capture_fraction"] >= 0.80) else "FALSIFIED"
out["verdicts"] = v
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
