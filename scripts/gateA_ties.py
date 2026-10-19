#!/usr/bin/env python
"""Near-tie analysis for the 683-vs-684 question. CPU only; reads the completed run + frozen data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/scripts")
import mvqa_diag as md  # noqa: E402

RD = Path(sys.argv[1])
gt = {g["case_id"]: md.LETTERS.index(g["answer"].strip()) for g in
      json.load(open(f"{md.FROZEN}/data_identity/ground_truth.json"))["ground_truth"] if g["task_type"] == "mcq"}
frozen = {a["case_id"]: md.LETTERS.index(a["answer"].strip()) for a in
          json.load(open(f"{md.FROZEN}/predictions/results.json"))["answers"] if a["task_type"] == "mcq"}
base = {r["case_id"]: r for r in md.read_jsonl(RD / "forward_arms.jsonl") if r["arm"] == "base"}
frozen_err = {c for c in gt if frozen[c] != gt[c]}

info = {}
for c, r in base.items():
    p = md.softmax5(r["direct"]["lp"])
    order = sorted(range(5), key=lambda k: -p[k])
    info[c] = {"margin": p[order[0]] - p[order[1]], "top1": order[0], "top2": order[1], "p1": p[order[0]], "p2": p[order[1]]}

print("near-tie counts (direct variant, base arm):")
for t in (1e-6, 1e-3, 1e-2, 0.1):
    ties = [c for c in info if info[c]["margin"] < t]
    print(f"  margin < {t:g}: all={len(ties)}  among frozen-28-errors={sum(c in frozen_err for c in ties)}"
          f"  gold is one of the top-2 letters in {sum(gt[c] in (info[c]['top1'], info[c]['top2']) for c in ties)}/{len(ties)}")

print("\nThe 5 cases where the forced-stub argmax disagrees with the frozen (actually-generated) answer:")
dis = [c for c in info if info[c]["top1"] != frozen[c]]
for c in dis:
    i = info[c]
    print(f"  {c}: gold={md.LETTERS[gt[c]]} frozen={md.LETTERS[frozen[c]]} forward_top1={md.LETTERS[i['top1']]} "
          f"top2={md.LETTERS[i['top2']]} p1={i['p1']:.4f} p2={i['p2']:.4f} margin={i['margin']:.4f} "
          f"frozen_was_correct={frozen[c] == gt[c]} forward_top1_correct={i['top1'] == gt[c]} "
          f"is_one_of_28_errors={c in frozen_err}")
print("\nfor frozen errors: margin quantiles")
m = sorted(info[c]["margin"] for c in frozen_err)
print("  ", [round(x, 4) for x in m])
exact = [c for c in frozen_err if info[c]["margin"] < 1e-3]
print(f"\nfrozen errors with margin < 1e-3 (decision hangs on a numerical tie): {len(exact)} -> {sorted(exact)}")
print("for each: gold letter in forward top-2?", {c: gt[c] in (info[c]['top1'], info[c]['top2']) for c in exact})
print("(if the gold letter is the tied runner-up, a kernel/hardware change that breaks the tie the other way gives +1 correct)")
