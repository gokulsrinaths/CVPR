#!/usr/bin/env python
"""Appends the A37 jobs to experiments/job_registry.csv with sacct runtimes. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = [23709531, 23709532, 23709533, 23709535, 23717814]
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    if name.startswith("a37_judge"):
        ds, note, cfg = "open-ended held-out judge scoring", ("A37 judge scoring (Qwen2.5-VL-7B reproduction judge)" + ("; first start failed at an import, see docs/..._A37_note.md" if state.startswith("FAILED") else "")), "slurm/47_a37_judge.sbatch"
    else:
        ds, note, cfg = "open-ended held-out (175 cases)", "A37 released adapter, image arm " + name.split("_")[-1] + " (A30)", "slurm/46_a37_infer.sbatch"
    rows.append([jid, "2026-10-06" if jid != "23717814" else "2026-10-07", name, ds, "175 open cases", "-", "CPU" if part == "beartooth" else "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, cfg, "data/holdout886_open_*/run_output; analysis/a37_*", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
