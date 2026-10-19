#!/usr/bin/env python
"""A1 analysis (docs/preregistration_v1_amendment_A1.md). python retrieval_summarize.py RUN_DIR"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, mcnemar_exact, gold_from_train, shortest_letter, argmax, margin, LETTERS

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
run_dir = sys.argv[1]
T = gold_from_train()
cfg = json.load(open(os.path.join(run_dir, "config.json")))
tau = cfg["tau_m"]
plan = json.load(open(os.path.join(run_dir, "retrieval_plan.json")))
rows = read_jsonl(os.path.join(run_dir, "rows.jsonl"))
arm = collections.defaultdict(dict)
for r in rows:
    arm[r["arm"]][r["case_id"]] = r
ga = {"base": {}, "black": {}}
for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
    for r in read_jsonl(pth):
        if r["arm"] in ga:
            ga[r["arm"]][r["case_id"]] = r["direct"]["lp"]
ids = sorted(set(plan) & set(arm["m1"]) & set(arm["m1c"]) & set(arm["m1leak"]) & set(arm["m1black"]))
n = len(ids)
gold = {i: T[i]["answer"].strip() for i in ids}
short = {i: shortest_letter(T[i]) for i in ids}
ws = [i for i in ids if short[i] != gold[i]]
m0 = {i: LETTERS[argmax(ga["base"][i])] for i in ids}
m0b = {i: LETTERS[argmax(ga["black"][i])] for i in ids}
m0margin = {i: margin(ga["base"][i]) for i in ids}
low = [i for i in ids if m0margin[i] < tau]
errs = [i for i in ids if m0[i] != gold[i]]
pred = {a: {i: LETTERS[argmax(arm[a][i]["direct"]["lp"])] for i in ids} for a in ("m1", "m1c", "m1leak", "m1black")}
pred["m2"] = {i: (pred["m1"][i] if m0margin[i] < tau else m0[i]) for i in ids}


def cmp(p, base, subset):
    f = sum(p[i] == gold[i] and base[i] != gold[i] for i in subset)
    b = sum(p[i] != gold[i] and base[i] == gold[i] for i in subset)
    k = sum(p[i] == gold[i] for i in subset)
    k0 = sum(base[i] == gold[i] for i in subset)
    return {"n": len(subset), "correct": k, "baseline_correct": k0, "fixed": f, "broken": b, "net": f - b, "mcnemar_p": mcnemar_exact(f, b),
            "acc": round(k / len(subset), 4) if subset else None}


out = {"run_dir": run_dir, "n_cases": n, "tau_m": tau, "n_low_margin": len(low), "n_m0_errors": len(errs),
       "low_margin_captures_errors": sum(i in set(low) for i in errs)}
views = {"all": ids, "shortest_wrong_63": ws, "low_margin": low}
for a in ("m1", "m1c", "m1leak", "m2"):
    out[a] = {v: cmp(pred[a], m0, s) for v, s in views.items()}
out["m1black_vs_m0black_all"] = cmp(pred["m1black"], m0b, ids)
out["m0"] = {"acc": round(sum(m0[i] == gold[i] for i in ids) / n, 4), "wilson": wilson(sum(m0[i] == gold[i] for i in ids), n)}
out["m0black"] = {"acc": round(sum(m0b[i] == gold[i] for i in ids) / n, 4)}
# diagnostics
mod_ok = sum(all(T[x]["primary_modality"] == plan[i]["modality"] for x in plan[i]["m1"]) for i in ids) / n
sims = [s for i in ids for s in plan[i]["m1_sims"]]
comp = json.load(open(ROOT + "/data/train_pool/split_v1/components.json"))
H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}
NT = set(json.load(open(ROOT + "/data/train_pool/split_v1/n_test_ids.json")))
members = collections.defaultdict(list)
for cid, cc in comp.items():
    members[cc].append(cid)
with_partner = [i for i in ids if any((p not in H) for p in members[comp[i]])]
rec = sum(any(comp[x] == comp[i] for x in plan[i]["m1leak"]) for i in with_partner)
rec1 = sum(any(comp[x] == comp[i] for x in plan[i]["m1"]) for i in with_partner)
out["diagnostics"] = {"m1_same_modality_rate": round(mod_ok, 4), "m1_mean_top3_similarity": round(sum(sims) / len(sims), 4),
                      "test_mcq_with_same_source_partner": len(with_partner), "partner_in_top3_m1leak": rec, "partner_in_top3_m1": rec1,
                      "m1leak_cases_with_a_neighbour_in_N_T": sum(any(x in NT for x in plan[i]["m1leak"]) for i in ids)}
# verdicts (A1 section 3)
h4 = out["m1"]["all"]
out["verdicts"] = {
    "H4": "FALSIFIED" if (h4["net"] >= 8 and h4["mcnemar_p"] < 0.05) else ("SUPPORTED" if h4["net"] <= 4 else "INCONCLUSIVE (between +5 and +7 or p>=0.05)"),
    "H4c": "SUPPORTED" if abs(out["m1c"]["all"]["net"]) <= 4 else "FALSIFIED",
    "H5": "SUPPORTED" if out["m1"]["shortest_wrong_63"]["net"] <= 0 else "FALSIFIED",
    "H6": "SUPPORTED" if (len(errs) and out["low_margin_captures_errors"] / len(errs) >= 0.60) else "FALSIFIED",
    "H7": "FALSIFIED" if (out["m2"]["all"]["net"] >= 8 and out["m2"]["all"]["mcnemar_p"] < 0.05) else ("SUPPORTED" if out["m2"]["all"]["net"] <= 4 else "INCONCLUSIVE")}
json.dump(out, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
