"""Exploratory: benefit relative to the d=0.25 benefit, and the interpolated strength (linear in logit d) at which it falls below 20% / 10%."""
import glob
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))


def load(sub):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(H, "paper_data", sub, "*.json")))]


pmc3 = [r for r in load("a17_runs") if r["shortcut"] == "length" and r["family"] == "q3b"]
pmch = load("a19_runs") + load("a22_runs")
ok = [r for r in load("a21_runs") if r["shortcut"] == "length"]
logit = lambda p: math.log(p / (1 - p))


def series(runs, fam):
    rs = [r for r in runs if r["family"] == fam and r["shortcut"] == "length"]
    ds = sorted({r["dose"] for r in rs})
    g = {d: float(np.mean([r["g"]["base"]["g"] for r in rs if abs(r["dose"] - d) < 1e-9])) for d in ds}
    n = {d: len([r for r in rs if abs(r["dose"] - d) < 1e-9]) for d in ds}
    return ds, g, n


def cross(ds, g, thr):
    rel = {d: g[d] / g[0.25] for d in ds}
    for a, b in zip(ds, ds[1:]):
        if rel[a] >= thr > rel[b]:
            x = logit(a) + (rel[a] - thr) / (rel[a] - rel[b]) * (logit(b) - logit(a))
            return 1 / (1 + math.exp(-x)), a, b
    return None, None, None


for name, runs, fam in (("PMC-VQA Qwen-3B", pmc3, "q3b"), ("PMC-VQA Qwen-7B", pmch, "q7b"), ("PMC-VQA InternVL3-2B", pmch, "ivl2b"), ("A-OKVQA Qwen-3B", ok, "q3b"), ("A-OKVQA Qwen-7B", ok, "q7b"), ("A-OKVQA InternVL3-2B", ok, "ivl2b")):
    ds, g, n = series(runs, fam)
    rel = {d: round(100 * g[d] / g[0.25], 1) for d in ds}
    c20 = cross(ds, g, 0.20)
    c10 = cross(ds, g, 0.10)
    print(name, "| doses", ds, "| seeds", [n[d] for d in ds])
    print("   g", {d: round(g[d], 2) for d in ds}, "| relative % of g(0.25)", rel)
    print("   d at 20%% of g(0.25): %s (between %s and %s) | d at 10%%: %s (between %s and %s)" % tuple([None if c20[0] is None else round(c20[0], 3), c20[1], c20[2], None if c10[0] is None else round(c10[0], 3), c10[1], c10[2]]))
