#!/usr/bin/env python
"""Appends the A23/A24 jobs to experiments/job_registry.csv with sacct runtimes. Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = list(range(23546232, 23546239)) + [23547542] + list(range(23548165, 23548175))
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
NOTE = [("a24_smoke", "smoke test of the A24 code path; numbers not used"), ("a23_finalize", "CPU job that summarised the A23 runs and ran the frozen analysis"), ("a24_finalize", "CPU job that summarised the A24 runs and ran the frozen analysis"),
        ("a23_q3b", "A23 part B: second seed of the A-OKVQA lock-out, Qwen 3B (A30). B1, B2 supported"), ("a23_q7b", "A23 part C: Qwen 7B on A-OKVQA (A30, gradient checkpointing). C1-C3 supported"),
        ("a24_lenprior", "A24 M1: explicit length prior as bias expert, real Med-CMR shortcut (A30). E1 not met. See analysis/a24_results.md"), ("a24_mu", "A24 M2: mismatched-image uniformity, real Med-CMR shortcut (A30). E1 not met"),
        ("a24_both", "A24 M3: length prior and mismatched-image uniformity, real Med-CMR shortcut (A30). E1 not met (accuracy about 79%)")]
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    note = next((t for p, t in NOTE if name.startswith(p)), "A23-A24 job")
    rows.append([jid, "2026-10-05", name, "see config.json", "zeroshot+ft eval or eval only", "see config.json", "CPU" if part == "beartooth" else "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, "slurm/29,33-35 *.sbatch", "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
