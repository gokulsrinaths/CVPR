#!/usr/bin/env python
"""Appends the A29, A30 and A35 jobs to experiments/job_registry.csv with sacct runtimes. Idempotent. python registry_a29_a35.py [extra job ids]"""
import csv
import subprocess
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = list(range(23688800, 23688808)) + [23688820, 23688983, 23688984, 23696614] + list(range(23699053, 23699062)) + [int(x) for x in sys.argv[1:]]
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%28,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)


def meta(name):
    if name.startswith("a29_q3b"):
        return ("OmniMedVQA planted length shortcut", "E1 (2,000)", "A29 lock-out prediction run (Qwen-3B, A30); prediction not supported, see analysis/a29_results.md", "slurm/41_a29_a30.sbatch")
    if name.startswith("a30_smoke"):
        return ("Med-CMR UL, Qwen-7B", "smoke", "smoke test of the A30 training script; numbers not used", "slurm/42_a30_train.sbatch")
    if name.startswith("a30_ul7b"):
        return ("Med-CMR UL, Qwen-7B", "FRESH3000 + TEST711", "A30 gold-unlikelihood training of Qwen-7B (A30 GPU, gradient checkpointing)", "slurm/42_a30_train.sbatch")
    if name.startswith("a30_b"):
        return ("Med-CMR UL, Qwen-7B", "FRESH3000 blur ladder", "A30 blur evaluation of a saved 7B UL adapter (evaluation only)", "slurm/43_a30_blur.sbatch")
    if name.startswith("a30_finalize"):
        return ("Med-CMR UL, Qwen-7B", "analysis", "CPU job that ran the frozen A30 analysis", "slurm/44_finalize_a30.sbatch")
    if name.startswith("a35_smoke"):
        return ("occlusion evaluation", "smoke", "smoke test of the A35 evaluation; numbers not used", "slurm/45_a35_occl.sbatch")
    if name.startswith("a35"):
        return ("PMC-VQA / A-OKVQA planted length shortcut", "E1", "A35 centre vs border occlusion of a saved dose-response adapter (Qwen-3B, evaluation only)", "slurm/45_a35_occl.sbatch")
    return ("see config.json", "see config.json", "A29/A30/A35 job", "slurm/")


n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    ds, sp, note, cfg = meta(name)
    seed = name.split("_")[-1] if name.split("_")[-1].isdigit() else "see config.json"
    rows.append([jid, "2026-10-05", name, ds, sp, seed, "CPU" if part == "beartooth" else "A30:1", el, "0:0" if state.startswith("COMPLETED") else state, cfg, "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
