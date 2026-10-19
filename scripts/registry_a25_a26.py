#!/usr/bin/env python
"""Appends the A25/A26 jobs to experiments/job_registry.csv with sacct runtimes. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = list(range(23554078, 23554090)) + [23561590] + list(range(23562005, 23562014))
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
NOTE = [("a26_smoke", "smoke test of the A26 code path; numbers not used"), ("a25_finalize", "CPU job that summarised the A25 runs and ran the frozen analysis"), ("a26_finalize", "CPU job that summarised the A26 runs and ran the frozen analysis"),
        ("a25_q7b", "A25: second seed of Qwen 7B on A-OKVQA (A30, gradient checkpointing). D1, D2, L1 supported"), ("a25_ivl", "A25: second seed of InternVL3-2B on A-OKVQA (A30, gradient checkpointing). D1, D2 supported; L1 falsified (g(0.99) = +1.66, V* = 5)"),
        ("a25_q3b", "A25: third seed of Qwen 3B on A-OKVQA at 0.92/0.97/0.99 (A30). L1, L2 supported"),
        ("a26_ul", "A26 UL (gold-unlikelihood under a mismatched image), real Med-CMR shortcut (A30). F1, F2 met in 3/3 seeds; black-image effect inflated by the penalty, see analysis/a26_results.md"),
        ("a26_lpul", "A26 LP+UL, real Med-CMR shortcut (A30). Collapsed (accuracy under 1%)"), ("a26_both", "A26 LP+MU with two further seeds, real Med-CMR shortcut (A30). F1 in 4/5 seeds, F2 in 0/5")]
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    note = next((t for p, t in NOTE if name.startswith(p)), "A23-A24 job")
    rows.append([jid, "2026-10-05", name, "see config.json", "zeroshot+ft eval or eval only", "see config.json", "CPU" if part == "beartooth" else "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, "slurm/29,36-38 *.sbatch", "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
