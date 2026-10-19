#!/usr/bin/env python
"""A17 aggregate analysis. python dose_analysis.py [--stop-check] -> analysis/a17_dose_summary.json (or prints the batch-1 stopping check)."""
import glob
import json
import sys

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
CONF_SEEDS = [20261011, 20261012, 20261013]
DOSES = [0.25, 0.40, 0.55, 0.70, 0.85, 0.92]
LADDER = ["blur32", "blur16", "blur8", "blur4", "blur2", "base"]


def load(family, shortcut, seeds):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/experiments/pilot/dose_{family}_{shortcut}*_seed*_2*/summary_dose.json")):
        if "smoke" in f:
            continue
        s = json.load(open(f))
        if s["seed"] in seeds:
            out.append(s)
    return out


def rank(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(runs, key, sign, n=10000, seed=20261011):
    """one-sided permutation p-value of the pooled Spearman rho(dose, key): doses permuted within seeds."""
    rng = np.random.RandomState(seed)
    x = np.array([r["dose"] for r in runs])
    y = np.array([key(r) for r in runs], float)
    sd = np.array([r["seed"] for r in runs])
    obs = spearman(x, y)
    cnt = 0
    for _ in range(n):
        xp = x.copy()
        for s in set(sd):
            m = sd == s
            xp[m] = rng.permutation(x[m])
        rho = spearman(xp, y)
        cnt += (rho * sign >= obs * sign)
    return obs, (cnt + 1) / (n + 1)


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ok = [False] * len(ps)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k):
            ok[i] = True
        else:
            break
    return ok


gi = lambda r: r["g"]["base"]["g"]
if "--stop-check" in sys.argv:
    runs = load("q3b", "length", [CONF_SEEDS[0]])
    g = {r["dose"]: gi(r) for r in runs}
    print("batch-1 g_intact by dose:", {k: g[k] for k in sorted(g)})
    if 0.25 in g and 0.92 in g:
        print("separation g(0.25) - g(0.92) =", round(g[0.25] - g[0.92], 2), "-> CONTINUE" if g[0.25] - g[0.92] >= 5.0 else "-> STOP (pre-registered stopping rule)")
    sys.exit(0)

L = load("q3b", "length", CONF_SEEDS)
res = {"n_runs_length": len(L), "runs": []}
for r in L:
    res["runs"].append({"dose": r["dose"], "seed": r["seed"], "g_intact": gi(r), "acc": r["acc"]["base"], "acc_black": r["acc"]["black"], "V_star": r["V_star_blur"], "V_star_down": r["V_star_down"],
                        "median_Mt": r["median_logodds_margin_black"], "share_ge_0.99": r["share_black_margin_ge_0.99"], "override_intact": r["override_rate_vs_black"]["base"],
                        "pred_shortest": r.get("pred_equals_shortest"), "orig_acc": r["orig_E1"]["acc"], "orig_gap": r["orig_E1"]["gap"], "realized_E1": r["realized_rate_E1"], "LO_benefit": r.get("LO_image_benefit")})
V = {}
if len(L) >= 6:
    seeds = sorted({r["seed"] for r in L})
    tests = {}
    rho1, p1 = perm_p(L, gi, -1)
    rho2, p2 = perm_p(L, lambda r: r["V_star_blur"], +1)
    rho3, p3 = perm_p(L, lambda r: r["median_logodds_margin_black"], +1)
    rho4, p4 = perm_p(L, lambda r: r["override_rate_vs_black"]["base"], -1)
    hm = holm([p1, p2, p3, p4])
    per_seed = {}
    for s in seeds:
        rs = [r for r in L if r["seed"] == s]
        per_seed[s] = {"rho_g": spearman([r["dose"] for r in rs], [gi(r) for r in rs]), "rho_V": spearman([r["dose"] for r in rs], [r["V_star_blur"] for r in rs]),
                       "g_025": next((gi(r) for r in rs if r["dose"] == 0.25), None), "g_092": next((gi(r) for r in rs if r["dose"] == 0.92), None),
                       "V_025": next((r["V_star_blur"] for r in rs if r["dose"] == 0.25), None), "V_092": next((r["V_star_blur"] for r in rs if r["dose"] == 0.92), None)}
    V["H1"] = {"rho": round(rho1, 3), "perm_p": p1, "holm_ok": hm[0], "holds": bool(rho1 <= -0.70 and hm[0] and all(v["rho_g"] <= -0.70 for v in per_seed.values()) and all((v["g_025"] or -99) >= 8.0 and (v["g_092"] if v["g_092"] is not None else 99) <= 2.0 for v in per_seed.values()))}
    V["H2"] = {"rho": round(rho2, 3), "perm_p": p2, "holm_ok": hm[1], "holds": bool(rho2 >= 0.70 and hm[1] and all((v["V_025"] or 99) <= 6 and v["V_092"] == 7 for v in per_seed.values()))}
    # H3: text margin tracks dose; leave-one-seed-out comparison of predictors of g_intact
    errs = {"dose": [], "Mt": []}
    for s in seeds:
        tr = [r for r in L if r["seed"] != s]
        te = [r for r in L if r["seed"] == s]
        for nm, f in (("dose", lambda r: r["dose"]), ("Mt", lambda r: r["median_logodds_margin_black"])):
            a, b = np.polyfit([f(r) for r in tr], [gi(r) for r in tr], 1)
            errs[nm] += [(a * f(r) + b - gi(r)) ** 2 for r in te]
    rmse = {k: float(np.sqrt(np.mean(v))) for k, v in errs.items()}
    V["H3"] = {"rho_dose_Mt": round(rho3, 3), "perm_p": p3, "holm_ok": hm[2], "rmse_dose": round(rmse["dose"], 3), "rmse_Mt": round(rmse["Mt"], 3), "holds_a": bool(rho3 >= 0.70 and hm[2]),
               "holds_b": bool(rmse["Mt"] <= 1.25 * rmse["dose"])}
    ov = {}
    for r in L:
        if r["dose"] <= 0.55:
            o = [r["override_rate_vs_black"][a] for a in LADDER]
            ov[f'{r["dose"]}|{r["seed"]}'] = round(spearman(range(6), o), 3)
    V["H4"] = {"rho_override_dose": round(rho4, 3), "perm_p": p4, "holm_ok": hm[3], "ladder_rho_low_doses": ov, "holds": bool(rho4 <= -0.70 and hm[3] and all(v >= 0.8 for v in ov.values()))}
    d = [r for r in L if r["V_star_down"] is not None]
    if d:
        V["H5_down_ladder"] = {"V025": [r["V_star_down"] for r in d if r["dose"] == 0.25], "V092": [r["V_star_down"] for r in d if r["dose"] == 0.92],
                               "holds": bool(all(r["V_star_down"] <= 6 for r in d if r["dose"] == 0.25) and all(r["V_star_down"] == 7 for r in d if r["dose"] == 0.92) and len(d) >= 2)}
    V["per_seed"] = per_seed
P = load("q3b", "position", [CONF_SEEDS[0]])
if len(P) >= 3:
    g = {r["dose"]: gi(r) for r in P}
    vs = {r["dose"]: r["V_star_blur"] for r in P}
    V["P_position"] = {"g": {k: g[k] for k in sorted(g)}, "V_star": {k: vs[k] for k in sorted(vs)},
                       "P1": bool(g.get(0.92, 99) <= 2.0), "P2": bool(g.get(0.25, -99) >= 8.0), "P3": bool(g.get(0.25, -99) >= g.get(0.55, 99) >= g.get(0.92, 99) - 1.0), "P4": bool(vs.get(0.92) == 7)}
res["verdicts"] = V
json.dump(res, open(ROOT + "/analysis/a17_dose_summary.json", "w"), indent=1)
print(json.dumps(V, indent=1))
for r in sorted(res["runs"], key=lambda r: (r["seed"], r["dose"])):
    print(r)
