#!/usr/bin/env python
"""Appends the A41 resubmission and the A44 jobs to experiments/job_registry.csv with sacct runtimes and states. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = list(range(24669503, 24669513))
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    if name.startswith("a41"):
        ds, sp, cfg, note = "six multiple-choice benchmarks", "zero-shot, true/black/told arms", "slurm/50_a41_zeroshot.sbatch", "A41 zero-shot Qwen2.5-VL-3B (resubmission of the cancelled first start; complete, 24,000 rows)"
        outp = "experiments/pilot/a41_3b_*"
    else:
        ds, sp, cfg = "Med-CMR training subset", "711 held-out MCQs, true/black", "slurm/51_a44_subset.sbatch"
        note = "A44 learning curve of the shortcut: Qwen2.5-VL-3B, A9 recipe, " + ("smoke test (numbers unused)" if "smoke" in name else name.replace("a44_", "").replace("_s3", " seed 20261003").replace("_s4", " seed 20261004"))
        outp = "experiments/pilot/a44_3b_*"
    rows.append([jid, "2026-10-18", name, ds, sp, "-", "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, cfg, outp, note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
