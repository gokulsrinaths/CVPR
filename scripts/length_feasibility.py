#!/usr/bin/env python
"""Option-length structure of the 711 test MCQs (CPU only, no model).

Part A (label-free, used to DESIGN a length-equalising perturbation): how unequal are the five options, and how much text a
truncate-to-shortest perturbation would destroy.
Part B (uses test labels; DIAGNOSTIC / exploratory, motivated by Gate A, see docs/preregistration_v1.md section 1): how the gold
option differs from distractors in length and simple surface cues, and baseline accuracy by length-gap bin.
Nothing here tunes a method. Writes analysis/length_feasibility.json and .md
"""
import csv
import json
import math
import re
import statistics as st
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
cases = json.load(open(ROOT + "/results/reproduction_frozen/data_identity/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
feat = {r["case_id"]: r for r in csv.DictReader(open(ROOT + "/analysis/mcq_all711_features.csv", encoding="utf-8"))}
LAB = re.compile(r"^\s*[A-E]\)\s*")
HEDGE = {"generally", "typically", "likely", "may", "might", "possibly", "suggests", "suggesting", "suggest", "usually", "often",
         "could", "probable", "probably", "potentially", "indicating", "consistent", "appears", "apparent", "suggestive"}


def words(t):
    return t.split()


def sign_test_p(a, b):
    """Exact two-sided sign test on paired values; ties dropped."""
    pos = sum(1 for x, y in zip(a, b) if x > y)
    neg = sum(1 for x, y in zip(a, b) if x < y)
    n = pos + neg
    if n == 0:
        return {"n": 0, "gold_greater": 0, "gold_smaller": 0, "p": 1.0}
    k = min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))
    return {"n": n, "gold_greater": pos, "gold_smaller": neg, "p": p}


rows = []
for c in cases:
    if c.get("task_type") != "mcq" or c["case_id"] not in feat:
        continue
    f = feat[c["case_id"]]
    opts = {o["label"]: LAB.sub("", o["text"]).strip() for o in c["options"]}
    labs = sorted(opts)
    gold, pred = f["ground_truth"].strip(), f["parsed_prediction"].strip()
    rows.append({"id": c["case_id"], "opts": opts, "labs": labs, "gold": gold, "pred": pred, "ok": gold == pred})
n = len(rows)
assert n == 711, n
out = {"n_mcq": n, "label_free": {}, "label_using_DIAGNOSTIC": {}}

# ---------------- Part A: label-free
mins, maxs, ratios, cvs, keep, short_mins = [], [], [], [], [], []
for r in rows:
    w = [len(words(r["opts"][l])) for l in r["labs"]]
    r["w"] = w
    r["ch"] = [len(r["opts"][l]) for l in r["labs"]]
    mn, mx = min(w), max(w)
    mins.append(mn)
    maxs.append(mx)
    ratios.append(mx / mn)
    cvs.append(st.pstdev(w) / st.mean(w))
    keep.append(sum(mn / x for x in w) / len(w))
    short_mins.append(mn)
A = out["label_free"]
A["words_per_option_median"] = st.median([x for r in rows for x in r["w"]])
A["words_per_option_iqr"] = [st.quantiles([x for r in rows for x in r["w"]], n=4)[0], st.quantiles([x for r in rows for x in r["w"]], n=4)[2]]
A["max_over_min_words_median"] = round(st.median(ratios), 3)
for t in (1.25, 1.5, 2.0):
    A["frac_cases_max_over_min_gt_" + str(t)] = round(sum(x > t for x in ratios) / n, 3)
A["mean_cv_of_option_words"] = round(st.mean(cvs), 3)
A["frac_options_already_within_20pct_of_each_other"] = round(sum(1 for r in rows if max(r["w"]) <= 1.2 * min(r["w"])) / n, 3)
A["truncate_to_shortest_mean_fraction_of_words_kept"] = round(st.mean(keep), 3)
A["truncate_to_shortest_min_words_median"] = st.median(short_mins)
A["frac_cases_shortest_option_under_5_words"] = round(sum(m < 5 for m in mins) / n, 3)
A["frac_cases_shortest_option_under_8_words"] = round(sum(m < 8 for m in mins) / n, 3)

# ---------------- Part B: label-using, diagnostic
B = out["label_using_DIAGNOSTIC"]
B["note"] = "Uses test labels. Descriptive only; seen on the test set before the preregistration, so exploratory."


def rule_acc(key):
    uniq = tied = ok = ok_t = 0
    for r in rows:
        vals = r[key]
        m = min(vals)
        idx = [i for i, v in enumerate(vals) if v == m]
        if len(idx) == 1:
            uniq += 1
            ok += (r["labs"][idx[0]] == r["gold"])
        else:
            tied += 1
            ok_t += (r["gold"] in [r["labs"][i] for i in idx])
    return {"unique_shortest_cases": uniq, "unique_shortest_is_gold": ok, "acc_on_unique": round(ok / max(uniq, 1), 4),
            "tied_shortest_cases": tied, "gold_among_tied": ok_t}


B["shortest_by_words"] = rule_acc("w")
B["shortest_by_chars"] = rule_acc("ch")
rank = {}
for r in rows:
    g = r["w"][r["labs"].index(r["gold"])]
    rk = 1 + sum(1 for v in r["w"] if v < g)
    rank[rk] = rank.get(rk, 0) + 1
B["gold_length_rank_by_words_1_shortest"] = dict(sorted(rank.items()))
gw, dw, gr, gc, dc, gh, dh = [], [], [], [], [], [], []
for r in rows:
    i = r["labs"].index(r["gold"])
    others = [j for j in range(len(r["labs"])) if j != i]
    gw.append(r["w"][i])
    dw.append(st.mean(r["w"][j] for j in others))
    gr.append(r["w"][i] / st.mean(r["w"][j] for j in others))
    txt = [r["opts"][l] for l in r["labs"]]
    gc.append(txt[i].count(","))
    dc.append(st.mean(txt[j].count(",") for j in others))
    hr = lambda s: sum(1 for x in re.findall(r"[a-z]+", s.lower()) if x in HEDGE)
    gh.append(hr(txt[i]))
    dh.append(st.mean(hr(txt[j]) for j in others))
    r["ratio"] = gr[-1]
B["gold_words_median"] = st.median(gw)
B["mean_distractor_words_median"] = round(st.median(dw), 2)
B["gold_over_mean_distractor_ratio_quartiles"] = [round(x, 3) for x in st.quantiles(gr, n=4)]
B["sign_test_gold_vs_mean_distractor_words"] = sign_test_p(gw, dw)
B["commas_gold_mean"] = round(st.mean(gc), 3)
B["commas_distractor_mean"] = round(st.mean(dc), 3)
B["sign_test_commas"] = sign_test_p(gc, dc)
B["hedge_words_gold_mean"] = round(st.mean(gh), 3)
B["hedge_words_distractor_mean"] = round(st.mean(dh), 3)
B["sign_test_hedge_words"] = sign_test_p(gh, dh)
bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 99)]
dose = []
for lo, hi in bins:
    sel = [r for r in rows if lo <= r["ratio"] < hi]
    k = sum(r["ok"] for r in sel)
    dose.append({"gold_over_mean_distractor_length": [lo, hi if hi < 99 else None], "n": len(sel), "baseline_correct": k,
                 "acc": round(k / len(sel), 4) if sel else None})
B["baseline_accuracy_by_gold_length_ratio_bin"] = dose

json.dump(out, open(ROOT + "/analysis/length_feasibility.json", "w"), indent=2)
lines = ["# Option-length structure of the 711 test MCQs", "",
         "CPU only. Part A is label-free (used to design a length-equalising perturbation). Part B uses test labels and is",
         "diagnostic/exploratory (see `docs/preregistration_v1.md` section 1). Raw numbers: `analysis/length_feasibility.json`.", "",
         "## Part A: label-free", ""]
for k, v in A.items():
    lines.append("- " + k + ": " + json.dumps(v))
lines += ["", "## Part B: label-using, diagnostic", ""]
for k, v in B.items():
    lines.append("- " + k + ": " + json.dumps(v))
open(ROOT + "/analysis/length_feasibility.md", "w").write("\n".join(lines) + "\n")
print(json.dumps(out, indent=1))
