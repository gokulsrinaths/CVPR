#!/usr/bin/env python
"""Appends the A40 and A41 jobs to experiments/job_registry.csv with sacct runtimes and states. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = [24544206, 24544207, 24544208, 24545584, 24545585, 24546370]
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    amend = "A41" if name.startswith("a41") else "A40"
    if state.startswith("COMPLETED"):
        note = f"{amend} zero-shot " + ("smoke test (numbers unused)" if "smoke" in name else "Qwen2.5-VL-7B on six benchmarks; result in analysis/a40_interim.md")
    else:
        note = f"{amend} job cancelled before producing results (the paper was finalised without it; see docs/preregistration_v1_amendment_{amend}_note.md)"
    rows.append([jid, "2026-10-12", name, "six multiple-choice benchmarks", "zero-shot, true/black/told arms", "-", "A30:1" if part == "beartooth-gpu" else part, el, "0:0" if state.startswith("COMPLETED") else state, "slurm/49_a40_zeroshot.sbatch" if amend == "A40" else "slurm/50_a41_zeroshot.sbatch", "experiments/pilot/a40_*" if amend == "A40" else "none", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
