#!/usr/bin/env python
"""A15 Part 3: visual-strength factorial (blur) and C3. python a15_factorial.py -> analysis/a15_factorial_summary.json"""
import glob
import json
import math

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
P = ROOT + "/experiments/pilot/"
FAM = {"q3b": "Qwen 3B", "q7b": "Qwen 7B", "ivl2b": "InternVL3-2B"}
MEDRUN = {"q3b": ("ft_3b_s0_22569048_*", "3b"), "q7b": ("ft_7b_s0_22569049_*", "7b"), "ivl2b": ("ft3_ivl2b_s0_22584009_*", "ivl2b")}
PMCRUN = {"q3b": "pmc_q3b_s0_22584006_*", "q7b": "pmc_q7b_s0_22584007_*", "ivl2b": "pmc_ivl2b_s0_22584008_*"}
LETP, LETM = "ABCD", "ABCDE"


def rows(pat, fname):
    f = [x for x in glob.glob(P + pat + "/" + fname) if "smoke" not in x]
    return [json.loads(x) for x in open(sorted(f)[-1])] if f else []


def collect(rs, grp, K, LET):
    t = {}
    for r in rs:
        if r["group"] == grp:
            t.setdefault(r["arm"], {})[r["case_id"]] = max(range(K), key=lambda i: r["direct"]["lp"][i]), LET.index(r["gold"])
    return t


def acc_and_override(t, arms, K):
    ids = set.intersection(*[set(t[a]) for a in arms])
    ids = sorted(ids)
    out = {}
    for a in arms:
        out[a] = {"acc": round(100 * sum(t[a][i][0] == t[a][i][1] for i in ids) / len(ids), 2),
                  "override_vs_black": round(100 * sum(t[a][i][0] != t["black"][i][0] for i in ids) / len(ids), 2)}
    return out, len(ids)


res = {}
for fam, name in FAM.items():
    # PMC original
    a = rows("a15pmc_%s_orig_2*" % fam, "rows_a15.jsonl")
    t = collect(a, "E1", 4, LETP)
    arms = [x for x in ("black", "blur24", "blur8", "base") if x in t]
    pm, n1 = acc_and_override(t, arms, 4) if len(arms) == 4 else ({}, 0)
    # planted
    b = rows("a15pmc_%s_implant_2*" % fam, "rows_a15.jsonl")
    t = collect(b, "E1", 4, LETP)
    arms = [x for x in ("black", "blur24", "blur8", "base") if x in t]
    pl, n2 = acc_and_override(t, arms, 4) if len(arms) == 4 else ({}, 0)
    # Med-CMR
    run, key = MEDRUN[fam]
    c = rows("a15med_%s_2*" % key, "rows_a15.jsonl")
    t = collect(c, "fresh", 5, LETM)
    arms = [x for x in ("black", "blur24", "blur8", "base") if x in t]
    mc, n3 = acc_and_override(t, arms, 5) if len(arms) == 4 else ({}, 0)
    res[name] = {"PMC_orig": pm, "Planted": pl, "Med-CMR": mc, "n": [n1, n2, n3]}
    print(name, json.dumps(res[name]), flush=True)


def g(d, a):
    return d[a]["acc"] - d["black"]["acc"]


v = {}
ok = all(res[n][k] for n in res for k in ("PMC_orig", "Planted", "Med-CMR"))
v["complete"] = ok
if ok:
    v["F1_monotone_PMC"] = {n: bool(g(res[n]["PMC_orig"], "blur24") < g(res[n]["PMC_orig"], "blur8") < g(res[n]["PMC_orig"], "base")) for n in res}
    v["F2_locked_medcmr"] = {n: bool(all(abs(g(res[n]["Med-CMR"], a)) <= 1.0 for a in ("blur24", "blur8", "base"))) for n in res}
    v["F2_locked_planted"] = {n: bool(all(abs(g(res[n]["Planted"], a)) <= 1.0 for a in ("blur24", "blur8", "base"))) for n in res}
    sl = lambda d: g(d, "base") - g(d, "blur24")
    v["F3_interaction"] = {n: bool(sl(res[n]["Planted"]) <= 0 or sl(res[n]["PMC_orig"]) >= 3 * sl(res[n]["Planted"])) for n in res}
    v["slopes_true_minus_blur24"] = {n: {"PMC_orig": round(sl(res[n]["PMC_orig"]), 2), "Planted": round(sl(res[n]["Planted"]), 2), "Med-CMR": round(sl(res[n]["Med-CMR"]), 2)} for n in res}
res["verdicts"] = v
print(json.dumps(v))
json.dump(res, open(ROOT + "/analysis/a15_factorial_summary.json", "w"), indent=1)
