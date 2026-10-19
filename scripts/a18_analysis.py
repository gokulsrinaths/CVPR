#!/usr/bin/env python
"""A18 analyses. python a18_analysis.py contrastive | verdicts
contrastive: training-free test-time contrastive scoring on the naive A17 runs (CPU). verdicts: M-A..M-G from run summaries."""
import glob
import json
import math
import sys

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
P = ROOT + "/experiments/pilot/"
LET = "ABCD"
SEEDS = [20261011, 20261012, 20261013]


def lsm(l):
    m = max(l)
    z = m + math.log(sum(math.exp(x - m) for x in l))
    return [x - z for x in l]


def read(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def arm_table(rows, group):
    t = {}
    for r in rows:
        if r["group"] == group:
            t.setdefault(r["arm"], {})[r["case_id"]] = r
    return t


def runs_of(pattern):
    return [d for d in sorted(glob.glob(P + pattern)) if "smoke" not in d]


def summ(d):
    return json.load(open(d + "/summary_dose.json"))


def contrastive():
    out = {}
    for d in runs_of("dose_q3b_length*_seed*_2*"):
        s = json.load(open(d + "/config.json"))
        if s["seed"] not in SEEDS or s["dose"] not in (0.25, 0.55, 0.70, 0.85, 0.92):
            continue
        res = {}
        for stage, grp in (("ft_planted", "E1"), ("ft_orig", "E1")):
            t = arm_table(read(f"{d}/rows_{stage}.jsonl"), grp)
            ids = sorted(set(t["base"]) & set(t["black"]))
            gold = {i: LET.index(t["base"][i]["gold"]) for i in ids}
            acc = lambda f: 100 * float(np.mean([f(i) == gold[i] for i in ids]))
            lt = {i: lsm(t["base"][i]["direct"]["lp"]) for i in ids}
            lb = {i: lsm(t["black"][i]["direct"]["lp"]) for i in ids}
            r = {"acc_true": round(acc(lambda i: int(np.argmax(lt[i]))), 2), "acc_black": round(acc(lambda i: int(np.argmax(lb[i]))), 2)}
            for lam in (0.5, 1.0):
                r[f"acc_contrastive_{lam}"] = round(acc(lambda i, lam=lam: int(np.argmax(np.array(lt[i]) - lam * np.array(lb[i])))), 2)
            res[stage] = r
        out[f'{s["dose"]}|{s["seed"]}'] = res
    v = {}
    sel = {k: x for k, x in out.items() if k.startswith("0.92|")}
    if len(sel) == 3:
        ok = True
        for k, x in sel.items():
            best = max(x["ft_planted"]["acc_contrastive_0.5"], x["ft_planted"]["acc_contrastive_1.0"])
            gc = best - x["ft_planted"]["acc_black"]
            imp = max(x["ft_orig"]["acc_contrastive_0.5"], x["ft_orig"]["acc_contrastive_1.0"]) - x["ft_orig"]["acc_true"]
            ok &= (gc < 5.0 and imp < 8.0)
            x["g_contrastive_best"] = round(gc, 2)
            x["orig_improvement_best"] = round(imp, 2)
        v["M_G_training_free_does_not_rescue"] = bool(ok)
    json.dump({"runs": out, "verdicts": v}, open(ROOT + "/analysis/a18_contrastive_summary.json", "w"), indent=1)
    print(json.dumps(v))
    for k, x in sorted(out.items()):
        print(k, x)


def verdicts():
    V, R = {}, {}
    naive = {(s["dose"], s["seed"]): s for s in (summ(d) for d in runs_of("dose_q3b_length*_seed*_2*/") if glob.glob(d + "/summary_dose.json")) if s["seed"] in SEEDS}
    poe = {(s["dose"], s["seed"]): s for s in (summ(d) for d in runs_of("rescue_poe_q3b_length*_seed*_2*/") if glob.glob(d + "/summary_dose.json")) if True}
    ctl = {(s["dose"], s["seed"]): s for s in (summ(d) for d in runs_of("rescue_control_q3b_length*_seed*_2*/") if glob.glob(d + "/summary_dose.json"))}
    ga = lambda s: s["g"]["base"]
    # M-A, M-B
    ma, mb = [], []
    for sd in SEEDS:
        if (0.92, sd) in poe and (0.92, sd) in naive:
            g = ga(poe[(0.92, sd)])
            ma.append(bool(g["g"] >= 5.0 and g["mcnemar_p"] < 0.05))
            mb.append(bool(poe[(0.92, sd)]["orig_E1"]["acc"] - naive[(0.92, sd)]["orig_E1"]["acc"] >= 8.0))
            R[f"seed{sd}"] = {"poe_g": g["g"], "poe_p": g["mcnemar_p"], "poe_orig_acc": poe[(0.92, sd)]["orig_E1"]["acc"], "naive_orig_acc": naive[(0.92, sd)]["orig_E1"]["acc"], "poe_planted_acc": poe[(0.92, sd)]["acc"]["base"],
                              "poe_V_star": poe[(0.92, sd)]["V_star_blur"], "naive_g": ga(naive[(0.92, sd)])["g"]}
    if len(ma) == 3:
        V["M_A_visual_control_restored"] = all(ma)
        V["M_B_cue_free_accuracy_recovered"] = all(mb)
    s0 = SEEDS[0]
    if (0.92, s0) in poe and (0.92, s0) in ctl:
        V["M_D_attribution"] = bool(ga(poe[(0.92, s0)])["g"] - ga(ctl[(0.92, s0)])["g"] >= 4.0)
        R["control_g"] = ga(ctl[(0.92, s0)])["g"]
    if (0.70, s0) in poe and (0.70, s0) in naive:
        V["M_E_intermediate_dose"] = bool(ga(poe[(0.70, s0)])["g"] - ga(naive[(0.70, s0)])["g"] >= 3.0)
        R["d070"] = {"poe_g": ga(poe[(0.70, s0)])["g"], "naive_g": ga(naive[(0.70, s0)])["g"]}
    # M-F: Med-CMR PoE (summary_a12.json from a12_summarize.py m2)
    mf = []
    for d in runs_of("a18_medcmr_poe_3b_*"):
        if not glob.glob(d + "/summary_a12.json"):
            continue
        s = json.load(open(d + "/summary_a12.json"))
        lo = s["on_a9_LO_set"]
        okf = bool(lo["diff_points"] >= 5.0 and lo["mcnemar_p"] is not None and lo["mcnemar_p"] < 0.05 and s["FRESH"]["acc_true"] >= 0.9548 - 0.02)
        mf.append(okf)
        R.setdefault("medcmr", []).append({"LO_effect": lo["diff_points"], "p": lo["mcnemar_p"], "acc": s["FRESH"]["acc_true"]})
    if len(mf) == 3:
        V["M_F_real_shortcut"] = bool(sum(mf) >= 2)
    try:
        V["M_G_training_free_does_not_rescue"] = json.load(open(ROOT + "/analysis/a18_contrastive_summary.json"))["verdicts"].get("M_G_training_free_does_not_rescue")
    except Exception:
        pass
    json.dump({"verdicts": V, "detail": R}, open(ROOT + "/analysis/a18_summary.json", "w"), indent=1)
    print(json.dumps(V, indent=1))
    print(json.dumps(R, indent=1))


{"contrastive": contrastive, "verdicts": verdicts}[sys.argv[1]]()
