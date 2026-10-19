#!/usr/bin/env python
"""Appends the A16-A20 jobs to experiments/job_registry.csv with scheduler-reported runtimes (sacct). Run on the cluster after all jobs have finished."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = (list(range(22973885, 22973889)) + list(range(22974483, 22974490)) + [22975405] + [22974490, 22974491] + list(range(22974819, 22974825)) + list(range(22983077, 22983092))
       + [22985841, 22985842, 22986587] + list(range(22986588, 22986595)) + list(range(22988013, 22988023)) + [22989492, 22989493] + list(range(22989631, 22989637)) + [23032336, 23035073, 23035074, 23035076])
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed", "-n", "-X", "-P"], text=True)
NOTE = [
    ("smk_", "smoke test of a code path; numbers not used"),
    ("a16_med7", "A16 7B seed replication, Med-CMR (A30, gradient checkpointing). R1-R4 supported"),
    ("a16_pmc7", "A16 7B seed replication, PMC-VQA (A30, gradient checkpointing). Q1-Q3 supported"),
    ("a16_omni", "A16 OmniMedVQA (A30, gradient checkpointing). O1,O3,O4 supported; O2 falsified"),
    ("a17_len", "A17 dose-response, length shortcut, Qwen 3B (A30). See analysis/a17_results.md"),
    ("a17_pos", "A17 second shortcut (answer position), Qwen 3B (A30). P1-P4 supported"),
    ("a18_tx", "A18 text-only bias expert (cross-fitted half) (A30)"),
    ("a18_medpoe", "A18 shortcut-rescue on Med-CMR (A30). M-F falsified (collapse to chance)"),
    ("a18_poe", "A18 shortcut-rescue on planted PMC-VQA (A30). M-A, M-B, M-E supported"),
    ("a18_ctl", "A18 control, product-of-experts loss with weight 0 (A30). M-D attribution supported"),
    ("a19_", "A19 held-out dose-response (A30, gradient checkpointing). See analysis/a19_results.md"),
    ("a20_", "A20 exploratory tempered bias expert on Med-CMR (A30). See analysis/a20_results.md"),
]
n = 0
for line in out.strip().splitlines():
    jid, name, state, el = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    note = next((t for p, t in NOTE if name.startswith(p)), "A16-A20 job")
    if state.startswith("CANCELLED"):
        note = "CANCELLED. " + note + (" (duplicate submission, cancelled before start)" if name.startswith("a18_") else "")
    rows.append([jid, "2026-10-04", name, "see config.json", "zeroshot+ft eval or eval only", "see config.json", "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, "slurm/25-28 *.sbatch", "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
