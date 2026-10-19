#!/usr/bin/env python
"""Appends the A38 jobs to experiments/job_registry.csv with sacct runtimes. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = [23718490, 23718491, 23718492]
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    gpu = "H100:1" if part == "mb-h100" else "A30:1"
    rows.append([jid, "2026-10-07", name, "open-ended held-out judge scoring (175 cases)", "judge-side counterfactuals", "-", gpu, el, "0:0" if state.startswith("COMPLETED") else state, "slurm/48_a38_judge.sbatch",
                 "analysis/a38_scores_*.json, a38_summary_*.json", "A38 reproduction judge " + name.split("_")[-1] + " (Qwen2.5-VL-Instruct), seven judge-side conditions"])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
