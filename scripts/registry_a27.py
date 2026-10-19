#!/usr/bin/env python
"""Appends the A27 jobs (blur-test evaluations of saved adapters, smoke test, finalize) to experiments/job_registry.csv with sacct runtimes. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = [23682507] + list(range(23682702, 23682709))
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
NOTE = [("a27_smoke", "smoke test of the A27 evaluation mode; numbers not used"), ("a27_finalize", "CPU job that ran the frozen A27 analysis"),
        ("a27_ul", "A27 blur test of a saved UL adapter (A30, evaluation only). Rule C supported; see analysis/a27_results.md"),
        ("a27_lpmu", "A27 blur test of a saved LP+MU adapter (A30, evaluation only); flat in sharpness"), ("a27_a9", "A27 blur test of the A9 reference adapter (A30, evaluation only); flat")]
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    note = next((t for p, t in NOTE if name.startswith(p)), "A27 job")
    rows.append([jid, "2026-10-05", name, "see config.json", "eval only (blurred true image)", "see config.json", "CPU" if part == "beartooth" else "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, "slurm/39-40 *.sbatch", "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
