#!/usr/bin/env python
"""
Unit test for mvqa_diag.summarize() on SYNTHETIC rows with known answers (CPU only, no model).
Every arm's letter distribution is peaked on the content of the frozen prediction, mapped through the option
rotation, so summarize() must reproduce exactly: accuracy 683/711 for every arm, agreement 1.0, zero permutation-SC
fixes/breaks, zero greedy flips. A second run perturbs some rows to check that fixes/breaks/flips are counted.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/scripts")
import mvqa_diag as md  # noqa: E402

frozen = {a["case_id"]: a["answer"].strip() for a in json.load(open(f"{md.FROZEN}/predictions/results.json"))["answers"]}
gt = {g["case_id"]: g["answer"].strip() for g in json.load(open(f"{md.FROZEN}/data_identity/ground_truth.json"))["ground_truth"]}
mcq = [c.case_id for c in md.load_mcq_cases()]
assert len(mcq) == 711


def lp_peak(j):
    v = [math.log(0.02)] * 5
    v[j] = math.log(0.92)
    return v


def write_run(d, mutate=None):
    d = Path(d)
    with open(d / "forward_arms.jsonl", "w") as f:
        for cid in mcq:
            for arm in md.ARMS:
                s = int(arm[4:]) if arm.startswith("perm") else 0
                idx = md.LETTERS.index(frozen[cid])
                j = 0 if arm in ("black", "mismatch") else (idx + s) % 5
                if mutate:
                    j = mutate(cid, arm, j, s)
                row = {"case_id": cid, "arm": arm, "direct": {"lp": lp_peak(j), "mass": 0.9, "top5": []},
                       "cue": {"lp": lp_peak(j), "mass": 0.9, "top5": []}}
                f.write(json.dumps(row) + "\n")
    with open(d / "greedy_repeat.jsonl", "w") as f:
        for cid in mcq:
            ans = frozen[cid]
            if mutate and mutate(cid, "greedy", None, 0) == "FLIP":
                ans = "A" if ans != "A" else "B"
            f.write(json.dumps({"case_id": cid, "parsed_answer": ans, "raw_decodes": [md.STUB + "\n</think>\n\nX"],
                                "forced_conclusion_path": False}) + "\n")


fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        fails.append(name)


n_correct = sum(frozen[c] == gt[c] for c in mcq)
with tempfile.TemporaryDirectory() as d:
    write_run(d)
    s = md.summarize(d)
    base = s["arms_direct"]["base"]
    check("base accuracy == frozen accuracy", base["accuracy"] == round(n_correct / 711, 4), str(base))
    check("base agreement with frozen == 1.0", base["agreement_with_frozen_prediction"] == 1.0)
    for a in ("perm1", "perm2", "perm3", "perm4"):
        check(f"{a} (mapped back) == frozen accuracy and agreement 1.0",
              s["arms_direct"][a]["accuracy"] == round(n_correct / 711, 4)
              and s["arms_direct"][a]["agreement_with_frozen_prediction"] == 1.0)
    psc = s["permutation_self_consistency_direct"]
    check("perm-SC: n=711, accuracy == frozen, 0 fixed, 0 broken",
          psc["n"] == 711 and psc["accuracy"] == round(n_correct / 711, 4) and psc["fixed_vs_base"] == 0 and psc["broken_vs_base"] == 0, str(psc))
    check("greedy: 0 flips, accuracy == frozen", s["greedy_repeat"]["flips_vs_frozen"] == 0 and s["greedy_repeat"]["correct"] == n_correct)
    check("greedy: raw-think-equals-stub counted 711", s["greedy_repeat"]["raw_think_equals_stub"] == 711)

# perturbation: make perm1..perm4 all choose the GOLD content for 5 baseline-wrong cases (-> 5 fixes), and all
# perms choose a wrong content for 3 baseline-right cases (-> 3 breaks); flip 2 greedy answers.
wrong = [c for c in mcq if frozen[c] != gt[c]][:5]
right = [c for c in mcq if frozen[c] == gt[c]][:3]
flip2 = set(right[:2])


def mutate(cid, arm, j, s):
    if arm == "greedy":
        return "FLIP" if cid in flip2 else None
    if arm in ("base", "black", "mismatch"):
        return j
    if cid in wrong:
        return (md.LETTERS.index(gt[cid]) + s) % 5
    if cid in right:
        bad = [k for k in range(5) if k != md.LETTERS.index(gt[cid])][0]
        return (bad + s) % 5
    return j


with tempfile.TemporaryDirectory() as d:
    write_run(d, mutate)
    s = md.summarize(d)
    psc = s["permutation_self_consistency_direct"]
    # the 4 perms outvote the single base ordering (4/5 of the mass), so the decision follows the perms
    check("perturbed: 5 fixes counted", psc["fixed_vs_base"] == 5, str(psc))
    check("perturbed: 3 breaks counted", psc["broken_vs_base"] == 3, str(psc))
    check("perturbed: 2 greedy flips counted", s["greedy_repeat"]["flips_vs_frozen"] == 2, str(s["greedy_repeat"]["flip_case_ids"]))

print("SUMMARY TEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
