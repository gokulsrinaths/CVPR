#!/usr/bin/env python
"""A17 per-run analysis. python dose_summarize.py RUN_DIR -> RUN_DIR/summary_dose.json
Endpoints: accuracy by arm, image benefit g(L) = acc(L) - acc(black) with exact McNemar and paired-bootstrap CI, visual veto boundary V*, text margin, overrides, shortcut pick rate."""
import json
import math
import os
import sys

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LET = "ABCD"
BLUR = ["blur32", "blur16", "blur8", "blur4", "blur2", "base"]      # ladder levels 1..6 (intact = 6)
DOWN = ["down8", "down16", "down32", "down64", "down128", "base"]
G_MIN, CI_MIN = 2.0, 1.0       # veto criterion (A17): g >= 2.0 points and bootstrap 95% lower bound >= 1.0
NONE_LEVEL = 7
run_dir = sys.argv[1]
cfg = json.load(open(os.path.join(run_dir, "config.json")))


def read_jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def lsm(l):
    m = max(l)
    z = m + math.log(sum(math.exp(x - m) for x in l))
    return [x - z for x in l]


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
            d.setdefault(r["arm"], {})[r["case_id"]] = r
    return d


rows = read_jsonl(os.path.join(run_dir, "rows_ft_planted.jsonl"))
val = table(rows, "VAL700")["base"]
vm = sorted((lambda p: p[0] - p[1])(sorted([math.exp(x) for x in lsm(r["direct"]["lp"])], reverse=True)) for r in val.values())
tau = vm[int(0.15 * len(vm))]
t = table(rows, "E1")
arms = sorted(t)
ids = sorted(set.intersection(*[set(t[a]) for a in arms]))
n = len(ids)
pred = {a: np.array([int(max(range(4), key=lambda k: t[a][i]["direct"]["lp"][k])) for i in ids]) for a in arms}
gold = np.array([LET.index(t["base"][i]["gold"]) for i in ids])
correct = {a: (pred[a] == gold).astype(int) for a in arms}
acc = {a: round(100 * float(correct[a].mean()), 2) for a in arms}
rng = np.random.RandomState(20261011)
BIDX = rng.randint(0, n, size=(2000, n))


def gstat(a):
    b = correct["black"]
    c = correct[a]
    fx = int(((c == 1) & (b == 0)).sum())
    br = int(((c == 0) & (b == 1)).sum())
    boot = 100 * (c[BIDX].mean(1) - b[BIDX].mean(1))
    return {"g": round(100 * float(c.mean() - b.mean()), 2), "ci95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)], "fixes": fx, "breaks": br, "mcnemar_p": mcnemar_exact(fx, br)}


def boundary(ladder):
    if not all(a in correct for a in ladder):
        return None
    for k, a in enumerate(ladder, start=1):
        s = gstat(a)
        if s["g"] >= G_MIN and s["ci95"][0] >= CI_MIN:
            return k
    return NONE_LEVEL


lp_b = [lsm(t["black"][i]["direct"]["lp"]) for i in ids]
mt = np.array([sorted(l, reverse=True)[0] - sorted(l, reverse=True)[1] for l in lp_b])
mb = np.array([(lambda p: p[0] - p[1])(sorted([math.exp(x) for x in l], reverse=True)) for l in lp_b])
strat = np.where(mb >= tau, "HI", np.where(mb >= 0.90, "MID", "LO"))
out = {"run_dir": run_dir, "family": cfg["family"], "shortcut": cfg["shortcut"], "dose": cfg["dose"], "seed": cfg["seed"], "n_E1": n, "realized_rate_TR": cfg["realized_rate_TR"], "realized_rate_E1": cfg["realized_rate_E1"],
       "tau_m": tau, "acc": acc, "g": {a: gstat(a) for a in arms if a != "black"}}
out["V_star_blur"] = boundary(BLUR)
out["V_star_down"] = boundary(DOWN)
out["median_logodds_margin_black"] = round(float(np.median(mt)), 3)
out["share_black_margin_ge_0.99"] = round(float((mb >= 0.99).mean()), 4)
out["share_black_margin_ge_0.50"] = round(float((mb >= 0.50).mean()), 4)
out["strata"] = {s: int((strat == s).sum()) for s in ("HI", "MID", "LO")}
lo = strat == "LO"
if lo.sum():
    out["LO_image_benefit"] = round(100 * float(correct["base"][lo].mean() - correct["black"][lo].mean()), 2)
out["override_rate_vs_black"] = {a: round(100 * float((pred[a] != pred["black"]).mean()), 2) for a in arms if a != "black"}
if cfg["shortcut"] == "length":
    # unique shortest option under the planted options is recomputed from the stored letters' options in the split; use pick-rate proxy via records
    REC = json.load(open(ROOT + "/data/aokvqa/split_a21/records.json"))
    R = {}
    for g in REC.values():
        R.update(g)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dose_plant import assign_u, length_options
    U = assign_u(json.load(open(ROOT + "/data/aokvqa/split_a21/ids.json"))["E1"])
    sh = []
    for k, i in enumerate(ids):
        ch = length_options(i, R[i]["choices"], LET.index(R[i]["gold"]), U[i], cfg["dose"])
        L = [len(c) for c in ch]
        sh.append(L.index(min(L)) if L.count(min(L)) == 1 else -1)
    out["pred_equals_shortest"] = round(float((pred["base"] == np.array(sh)).mean()), 4)
else:
    out["pred_equals_A"] = round(float((pred["base"] == 0).mean()), 4)
orig = table(read_jsonl(os.path.join(run_dir, "rows_ft_orig.jsonl")), "E1")
oids = sorted(set(orig["base"]) & set(orig["black"]))
og = lambda a: 100 * float(np.mean([int(max(range(4), key=lambda k: orig[a][i]["direct"]["lp"][k])) == LET.index(orig[a][i]["gold"]) for i in oids]))
out["orig_E1"] = {"acc": round(og("base"), 2), "acc_black": round(og("black"), 2), "gap": round(og("base") - og("black"), 2)}
tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
out["training"] = {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]}
json.dump(out, open(os.path.join(run_dir, "summary_dose.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("family", "shortcut", "dose", "seed", "realized_rate_E1", "acc", "V_star_blur", "median_logodds_margin_black", "override_rate_vs_black", "orig_E1")}, indent=1))
print({a: (v["g"], v["ci95"]) for a, v in out["g"].items()})
