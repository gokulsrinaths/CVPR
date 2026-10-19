#!/usr/bin/env python
"""A30 analysis: exactly rules H1-H3 and the reading rule of docs/preregistration_v1_amendment_A30.md. python a30_analysis.py -> analysis/a30_summary.json
Uses summary_a12.json (a12_summarize.py m2) of each training run, rows_ft.jsonl for the plausible-wrong-image arm, and rows_blur.jsonl of the A27-style blur evaluation."""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SEEDS = (20261051, 20261052)
LEVELS = ["blur32", "blur16", "blur8", "blur4", "blur2", "base"]


def accs(path, group, arms):
    P = {}
    for x in open(path):
        r = json.loads(x)
        if r["group"] == group and r["arm"] in arms:
            P.setdefault(r["arm"], {})[r["case_id"]] = (max(range(5), key=lambda k: r["direct"]["lp"][k]), r["gold"])
    return {a: 100 * sum(p == "ABCDE".index(g) for p, g in P[a].values()) / len(P[a]) for a in P}


out = {"seeds": {}}
for sd in SEEDS:
    d = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a26_medcmr_ul_7b_seed{sd}_2*") if "smoke" not in x]
    b = glob.glob(f"{ROOT}/experiments/pilot/a27_blur_ul7b{sd % 100}_2*/rows_blur.jsonl")
    if not d or not (glob.glob(d[0] + "/summary_a12.json")):
        out["seeds"][str(sd)] = {"incomplete": "training/summary"}
        continue
    s = json.load(open(d[0] + "/summary_a12.json"))
    al = s["FRESH"]["all"]
    acc = 100 * s["FRESH"]["acc_true"]
    A4 = accs(d[0] + "/rows_ft.jsonl", "fresh", ["base", "black", "mismatch_rand", "mismatch_nn"])
    h1 = bool(al["diff_points"] >= 3.0 and al["mcnemar_p"] < 0.025 and acc >= 88.0)
    h2 = bool(A4["base"] - A4["mismatch_nn"] >= 5.0)
    rec = {"gap_all": al["diff_points"], "p": al["mcnemar_p"], "fresh_acc": acc, "test_acc": 100 * s["TEST711"]["acc_true"], "acc_arms": {k: round(v, 2) for k, v in A4.items()},
           "true_minus_plausible_wrong": round(A4["base"] - A4["mismatch_nn"], 2), "H1": h1, "H2": h2}
    if b:
        A = accs(sorted(b)[-1], "fresh", LEVELS)
        seq = [A[l] for l in LEVELS]
        gap = seq[-1] - A4["black"]
        rec["blur_acc"] = {l: round(A[l], 2) for l in LEVELS}
        rec["H3_C"] = bool(seq[-1] - seq[0] >= 10.0 and all(seq[i + 1] - seq[i] >= -1.0 for i in range(len(seq) - 1)))
        rec["R8"] = round((A["blur8"] - A4["black"]) / gap, 3) if abs(gap) > 1e-9 else None
    out["seeds"][str(sd)] = rec
done = [v for v in out["seeds"].values() if "incomplete" not in v]
if len(done) == len(SEEDS):
    n1 = sum(v["H1"] for v in done)
    out["H1_seeds"] = n1
    out["verdict"] = ("UL transfers to 7B: image reliance restored in both seeds" if n1 == 2 else "mixed: image reliance restored in one of two seeds" if n1 == 1 else "UL does not transfer to 7B at these settings")
    out["H2_where_H1"] = {k: v["H2"] for k, v in out["seeds"].items() if v["H1"]}
    out["H3_C_seeds"] = {k: v.get("H3_C") for k, v in out["seeds"].items()}
    out["R8_seeds"] = {k: v.get("R8") for k, v in out["seeds"].items()}
json.dump(out, open(ROOT + "/analysis/a30_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
