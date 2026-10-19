#!/usr/bin/env python
"""Appends the A21/A22 jobs (runs, smoke tests, probes, finalize) to experiments/job_registry.csv with scheduler-reported runtimes (sacct). Idempotent."""
import csv
import subprocess

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
IDS = (list(range(23189275, 23189284)) + [23187187, 23242877, 23254336, 23256833, 23256834] + [23260407, 23260408, 23260410, 23260412, 23260413]
       + [23260415, 23260416, 23260418, 23260419, 23260420, 23260422, 23270223])
rows = list(csv.reader(open(ROOT + "/experiments/job_registry.csv")))
have = {r[0] for r in rows[1:]}
out = subprocess.check_output(["sacct", "-j", ",".join(str(i) for i in IDS), "--format=JobID,JobName%24,State,Elapsed,Partition", "-n", "-X", "-P"], text=True)
NOTE = [
    ("a21_smoke", "smoke test of the A21 code path (A-OKVQA, 640 training questions); numbers not used"),
    ("r6000_probe", "probe of the RTX PRO 6000 Blackwell GPU and CUDA 12.8 PyTorch environment; the first probe failed (sm_120 not supported by the shared CUDA 12.6 PyTorch), the second passed"),
    ("a22_smoke", "smoke test of the A22 code path on the RTX PRO 6000; numbers not used"),
    ("a21_q3b", "A21 non-medical dose-response (A-OKVQA, length shortcut), Qwen 3B (A30). R1, R3 supported; R2, R4 falsified. See analysis/a21_results.md"),
    ("a21_ivl2b", "A21 non-medical dose-response (A-OKVQA, length shortcut), InternVL3-2B (A30, gradient checkpointing). R1, R3 supported; R2, R4 falsified. See analysis/a21_results.md"),
    ("a22_h_", "A22 part H: higher strength on A-OKVQA, Qwen 3B (RTX PRO 6000, CUDA 12.8 environment). H1-H3 supported; lock-out at 0.99. See analysis/a22_results.md"),
    ("a22_p_", "A22 part P: position shortcut on A-OKVQA, Qwen 3B (RTX PRO 6000, CUDA 12.8 environment). P1, P2 supported; P3 falsified. See analysis/a22_results.md"),
    ("a22_s_", "A22 part S: second seed of the A19 held-out test (A30, gradient checkpointing). Rules A, B, C supported. See analysis/a22_results.md"),
    ("a2x_finalize", "CPU job that summarised the finished A21/A22 runs and ran the frozen analysis scripts"),
]
GPU = {"mb-r6000": "R6000:1", "beartooth-gpu": "A30:1", "beartooth": "CPU"}
n = 0
for line in out.strip().splitlines():
    jid, name, state, el, part = line.split("|")
    if jid in have or not jid.isdigit():
        continue
    note = next((t for p, t in NOTE if name.startswith(p)), "A21-A22 job")
    if state.startswith("CANCELLED"):
        note = "CANCELLED while pending (not needed). " + note
    elif state.startswith("FAILED"):
        note = "FAILED (reported). " + note
    rows.append([jid, "2026-10-04", name, "see config.json", "zeroshot+ft eval or eval only", "see config.json", GPU.get(part, part), el, "0:0" if state.startswith("COMPLETED") else state,
                 "slurm/29-32 *.sbatch", "experiments/pilot/ (by job id)", note])
    n += 1
csv.writer(open(ROOT + "/experiments/job_registry.csv", "w", newline="")).writerows(rows)
print("appended", n, "rows; total", len(rows), "bad widths", [i for i, x in enumerate(rows) if len(x) != 12])
