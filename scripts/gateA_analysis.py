#!/usr/bin/env python
"""
Gate A' analysis (CPU only). Reads the completed mvqa_diag run + frozen data; writes analysis/gateA_results.json.
Everything here follows docs/gate_A_prespec.md; the direct variant is the pre-specified primary (mean letter mass = 1.0),
the cue variant is reported as secondary and is never promoted after the fact.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/scripts")
import mvqa_diag as md  # noqa: E402

ROOT = md.ROOT
RD = Path(sys.argv[1])
gt = {g["case_id"]: md.LETTERS.index(g["answer"].strip()) for g in
      json.load(open(f"{md.FROZEN}/data_identity/ground_truth.json"))["ground_truth"] if g["task_type"] == "mcq"}
frozen = {a["case_id"]: md.LETTERS.index(a["answer"].strip()) for a in
          json.load(open(f"{md.FROZEN}/predictions/results.json"))["answers"] if a["task_type"] == "mcq"}
cases = {c.case_id: c for c in md.load_mcq_cases()}
rows = md.read_jsonl(RD / "forward_arms.jsonl")
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["case_id"]] = r
ids = [c for c in cases]
assert all(len(by[a]) == 711 for a in md.ARMS)


def mcnemar_p(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pick(arm, cid, variant):
    s = int(arm[4:]) if arm.startswith("perm") else 0
    lp = by[arm][cid][variant]["lp"]
    return md.orig_index(max(range(5), key=lambda k: lp[k]), s)


def perm_sc(cid, variant):
    prob = [0.0] * 5
    for arm in ("base", "perm1", "perm2", "perm3", "perm4"):
        s = int(arm[4:]) if arm.startswith("perm") else 0
        p = md.softmax5(by[arm][cid][variant]["lp"])
        for j in range(5):
            prob[md.orig_index(j, s)] += p[j] / 5
    return max(range(5), key=lambda k: prob[k])


def shortest(cid):
    t = [o.text for o in cases[cid].options]
    return min(range(5), key=lambda k: (len(t[k]), k))


out = {"run_dir": str(RD)}
for variant in ("direct", "cue"):
    base = {c: pick("base", c, variant) for c in ids}
    black = {c: pick("black", c, variant) for c in ids}
    mism = {c: pick("mismatch", c, variant) for c in ids}
    psc = {c: perm_sc(c, variant) for c in ids}
    short = {c: shortest(c) for c in ids}
    ok = lambda d: {c for c in ids if d[c] == gt[c]}
    B, K, M, P, S = ok(base), ok(black), ok(mism), ok(psc), {c for c in ids if short[c] == gt[c]}
    V = {}

    # --- image reliance (paired)
    V["image_reliance"] = {
        "base_correct": len(B), "black_correct": len(K), "mismatch_correct": len(M),
        "black: base_right_black_wrong (image helps)": len(B - K), "black: base_wrong_black_right (image hurts)": len(K - B),
        "black_vs_base_exact_mcnemar_p": round(mcnemar_p(len(B - K), len(K - B)), 4),
        "mismatch: base_right_mismatch_wrong": len(B - M), "mismatch: base_wrong_mismatch_right": len(M - B),
        "mismatch_vs_base_exact_mcnemar_p": round(mcnemar_p(len(B - M), len(M - B)), 4),
        "black_picks_equal_to_base": round(sum(black[c] == base[c] for c in ids) / 711, 4),
        "black_picks_equal_to_shortest": round(sum(black[c] == short[c] for c in ids) / 711, 4),
        "image_necessary_cases (base right, BOTH black and mismatch wrong)": len(B - K - M)}

    # --- strata: is the shortest option the answer?
    strata = {}
    for name, S_set in (("shortest_is_gold (n=%d)" % len(S), S), ("shortest_is_WRONG (n=%d)" % (711 - len(S)), set(ids) - S)):
        n = len(S_set)
        strata[name] = {"n": n, "base": len(B & S_set), "black": len(K & S_set), "mismatch": len(M & S_set),
                        "perm_self_consistency": len(P & S_set), "shortest_rule": len(S & S_set)}
    V["strata_by_shortest_option"] = strata
    V["length_only_control"] = {"shortest_option_rule_correct": len(S), "accuracy": round(len(S) / 711, 4)}

    # --- permutation self-consistency vs base (paired exact test) + what the changes look like
    fixed, broken = sorted(P - B), sorted(B - P)
    V["permutation_self_consistency"] = {
        "accuracy": round(len(P) / 711, 4), "correct": len(P), "base_correct": len(B),
        "fixed": len(fixed), "broken": len(broken), "net": len(fixed) - len(broken),
        "exact_mcnemar_p_two_sided": round(mcnemar_p(len(fixed), len(broken)), 4),
        "fixed_where_gold_is_shortest": sum(c in S for c in fixed),
        "broken_where_gold_is_shortest": sum(c in S for c in broken),
        "fixed_case_ids": fixed, "broken_case_ids": broken,
        "accuracy_beyond_shortest_rule_pp": round(100 * (len(P) - len(S)) / 711, 2)}
    for a in ("perm1", "perm2", "perm3", "perm4"):
        pa = {c for c in ids if pick(a, c, variant) == gt[c]}
        V[f"single_{a}"] = {"correct": len(pa), "fixed_vs_base": len(pa - B), "broken_vs_base": len(B - pa),
                            "exact_mcnemar_p": round(mcnemar_p(len(pa - B), len(B - pa)), 4)}

    # --- confidence (base arm): margin = top1 - top2 probability over the 5 letters
    marg, wrong_flag = {}, {}
    for c in ids:
        p = sorted(md.softmax5(by["base"][c][variant]["lp"]), reverse=True)
        marg[c] = p[0] - p[1]
        wrong_flag[c] = int(base[c] != gt[c])
    errs = [c for c in ids if wrong_flag[c]]
    frozen_errs = [c for c in ids if frozen[c] != gt[c]]
    rc = {}
    for t in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        fl = [c for c in ids if marg[c] < t]
        rc[str(t)] = {"flagged": len(fl), "flagged_pct": round(100 * len(fl) / 711, 1),
                      "base_errors_caught": f"{sum(wrong_flag[c] for c in fl)}/{len(errs)}",
                      "frozen_28_errors_caught": f"{sum(c in fl for c in frozen_errs)}/{len(frozen_errs)}",
                      "precision": round(sum(wrong_flag[c] for c in fl) / len(fl), 3) if fl else None}
    V["confidence_margin_risk_coverage"] = rc
    V["confidence_auroc_for_correctness"] = md.auroc([marg[c] for c in ids], [1 - wrong_flag[c] for c in ids])

    # --- reconstruction faithfulness: where does the forced-stub argmax disagree with the frozen prediction?
    dis = [c for c in ids if base[c] != frozen[c]]
    V["reconstruction_disagreements_with_frozen"] = {
        "n": len(dis), "case_ids": dis, "their_margins": [round(marg[c], 3) for c in dis],
        "median_margin_all_cases": round(sorted(marg.values())[355], 6)}
    out[variant] = V

# --- greedy repeat vs frozen (already 0 flips) and raw generation facts
g = md.read_jsonl(RD / "greedy_repeat.jsonl")
out["greedy_repeat"] = {"n": len(g), "flips_vs_frozen": sum(r["parsed_answer"] != md.LETTERS[frozen[r["case_id"]]] for r in g),
                        "forced_conclusion_path": sum(bool(r["forced_conclusion_path"]) for r in g),
                        "raw_equals_stub_plus_letter": sum(r["raw_decodes"][0].startswith(md.STUB + "\n</think>\n\n") for r in g),
                        "mean_seconds_per_case": round(sum(r["seconds"] for r in g) / len(g), 3)}
json.dump(out, open(f"{ROOT}/analysis/gateA_results.json", "w"), indent=2)
print(json.dumps(out, indent=1))
