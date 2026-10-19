#!/usr/bin/env python
"""A43: a frozen one-parameter hypothesis for the lock-out strength. Visual evidence V = mean over questions of log p(gold | true image) - log p(gold | black image) (nats, option letters normalised) of the d = 0.25 control model;
shortcut evidence S(d) = log(d (k-1) / (1-d)) (nats; zero at d = 1/k). Hypothesis H: lock-out occurs where S(d*) = lambda * V, with one constant lambda, calibrated once on Qwen-3B / PMC-VQA.
python a43_predict.py predict  -> analysis/a43_prediction.json   (no outcome of the new runs is used)
python a43_predict.py evaluate -> analysis/a43_summary.json      (after the new OmniMedVQA runs)"""
import glob
import json
import math
import sys

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
PILOT = ROOT + "/experiments/pilot"
K = 4
MODE = sys.argv[1]
TOL = 0.5


def S(d):
    return math.log(d * (K - 1) / (1 - d))


def d_of_S(s):
    e = math.exp(s)
    return e / (e + K - 1)


SERIES = {  # (benchmark, model): (glob prefix of the run directories, group of doses available)
    ("PMC-VQA", "Qwen-3B"): "dose_q3b_length", ("PMC-VQA", "Qwen-7B"): "dose_q7b_length", ("PMC-VQA", "InternVL3-2B"): "dose_ivl2b_length",
    ("A-OKVQA", "Qwen-3B"): "a21_q3b_length", ("A-OKVQA", "Qwen-7B"): "a21_q7b_length", ("A-OKVQA", "InternVL3-2B"): "a21_ivl2b_length",
    ("OmniMedVQA", "Qwen-3B"): "a29_q3b_length",
}


def runs(prefix, d):
    return [x for x in glob.glob(f"{PILOT}/{prefix}{d:.2f}_seed*_2*") if "smoke" not in x and glob.glob(x + "/summary_dose.json")]


def lsm(l):
    l = np.array(l, float)
    m = l.max()
    return l - (m + np.log(np.exp(l - m).sum()))


def visual_evidence(prefix):
    vs = []
    for d in runs(prefix, 0.25):
        P = {}
        for ln in open(d + "/rows_ft_planted.jsonl"):
            r = json.loads(ln)
            if r["group"] == "E1" and r["arm"] in ("base", "black"):
                P.setdefault(r["case_id"], {})[r["arm"]] = (lsm(r["direct"]["lp"]), "ABCD".index(r["gold"]))
        delta = [P[i]["base"][0][P[i]["base"][1]] - P[i]["black"][0][P[i]["black"][1]] for i in P if len(P[i]) == 2]
        vs.append(float(np.mean(delta)))
    return float(np.mean(vs)), len(vs)


def bracket(prefix):
    """highest dose with an unlocked seed, lowest dose at which every seed is locked (V* = none = 7); only doses above the control."""
    doses = sorted({float(x.split(prefix)[1][:4]) for x in glob.glob(f"{PILOT}/{prefix}*_seed*_2*") if "smoke" not in x and glob.glob(x + "/summary_dose.json")})
    lock = {}
    for d in doses:
        v = [json.load(open(x + "/summary_dose.json"))["V_star_blur"] for x in runs(prefix, d)]
        lock[d] = all(t == 7 for t in v)
    unl = [d for d in doses if not lock[d]]
    loc = [d for d in doses if lock[d]]
    # the bracket is [last unlocked dose below the first locked dose, first locked dose]
    first_locked = min(loc) if loc else None
    last_unl = max([d for d in unl if first_locked is None or d < first_locked], default=0.25)
    return last_unl, first_locked, lock


out = {"series": {}}
for key, prefix in SERIES.items():
    if key == ("OmniMedVQA", "Qwen-3B") and MODE == "predict":
        v, n = visual_evidence(prefix)
        out["series"][f"{key[0]}|{key[1]}"] = {"V": round(v, 4), "n_seeds": n}
        continue
    v, n = visual_evidence(prefix)
    lo, hi, lock = bracket(prefix)
    out["series"][f"{key[0]}|{key[1]}"] = {"V": round(v, 4), "n_seeds": n, "last_unlocked": lo, "first_locked": hi, "S_lo": round(S(lo), 3), "S_hi": round(S(hi), 3) if hi else None, "locked_by_dose": {str(d): b for d, b in lock.items()}}
cal = out["series"]["PMC-VQA|Qwen-3B"]
lam = 0.5 * (cal["S_lo"] + cal["S_hi"]) / cal["V"]
out["lambda"] = round(lam, 4)
out["calibration"] = "lambda = mean of S at the unlocked and locked dose of Qwen-3B / PMC-VQA, divided by its V"
for k, v in out["series"].items():
    v["S_pred"] = round(lam * v["V"], 3)
    v["d_star_pred"] = round(d_of_S(lam * v["V"]), 4)
if MODE == "predict":
    json.dump(out, open(ROOT + "/analysis/a43_prediction.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0)

pred = json.load(open(ROOT + "/analysis/a43_prediction.json"))
res = {}
for k, v in out["series"].items():
    if v.get("S_hi") is None:
        ok = v["S_pred"] >= v["S_lo"] - TOL
    else:
        ok = bool(v["S_lo"] - TOL <= v["S_pred"] <= v["S_hi"] + TOL)
    res[k] = {"V": v["V"], "S_pred": v["S_pred"], "d_star_pred": v["d_star_pred"], "bracket_d": [v["last_unlocked"], v["first_locked"]], "bracket_S": [v["S_lo"], v["S_hi"]], "inside_bracket_pm_0.5_nats": ok}
    print(k, json.dumps(res[k]), flush=True)
out["results"] = res
out["P1_AOKVQA_3B"] = res["A-OKVQA|Qwen-3B"]["inside_bracket_pm_0.5_nats"]
out["P2_three_of_four_other"] = bool(sum(res[k]["inside_bracket_pm_0.5_nats"] for k in ("PMC-VQA|Qwen-7B", "PMC-VQA|InternVL3-2B", "A-OKVQA|Qwen-7B", "A-OKVQA|InternVL3-2B")) >= 3)
out["P3_OmniMedVQA_3B_prospective"] = res["OmniMedVQA|Qwen-3B"]["inside_bracket_pm_0.5_nats"]
out["omni_prediction_unchanged_from_frozen_file"] = bool(abs(pred["series"]["OmniMedVQA|Qwen-3B"]["S_pred"] - out["series"]["OmniMedVQA|Qwen-3B"]["S_pred"]) < 1e-6)
json.dump(out, open(ROOT + "/analysis/a43_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("series", "results")}))
