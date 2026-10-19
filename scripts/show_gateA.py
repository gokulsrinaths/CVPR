import json

d = json.load(open("/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/analysis/gateA_results.json"))
dv, cv = d["direct"], d["cue"]
print("=== PRIMARY (direct variant) ===")
for k in ("image_reliance", "length_only_control", "strata_by_shortest_option"):
    print(k, json.dumps(dv[k], indent=1))
psc = {k: v for k, v in dv["permutation_self_consistency"].items() if k not in ("fixed_case_ids", "broken_case_ids")}
print("permutation_self_consistency", json.dumps(psc, indent=1))
print("fixed ids:", dv["permutation_self_consistency"]["fixed_case_ids"])
print("broken ids:", dv["permutation_self_consistency"]["broken_case_ids"])
print("single perms:", json.dumps({k: v for k, v in dv.items() if k.startswith("single_")}))
print("confidence AUROC:", dv["confidence_auroc_for_correctness"])
print("risk-coverage:", json.dumps(dv["confidence_margin_risk_coverage"], indent=1))
print("reconstruction disagreements:", dv["reconstruction_disagreements_with_frozen"])
print("=== SECONDARY (cue variant) ===")
print(json.dumps({k: v for k, v in cv["permutation_self_consistency"].items() if k not in ("fixed_case_ids", "broken_case_ids")}))
print(json.dumps(cv["image_reliance"]))
print("greedy:", d["greedy_repeat"])
