"""Exploratory: accuracy of the A26 UL runs (and references) with each image arm on FRESH3000, overall and by whether the correct option is the shortest."""
import glob
import json
import re

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}


def shortest(cid):
    L = [len(LAB.sub("", T[cid][l]).strip()) for l in "ABCDE"]
    return min(range(5), key=lambda k: (L[k], k))


RUNS = [("A9 Qwen-3B reference", "ft_3b_s0_22569048_*")]
for sd in (20261041, 20261042, 20261043):
    RUNS.append((f"A26 UL seed {sd % 100}", f"a26_medcmr_ul_3b_seed{sd}_2*"))
for sd in (20261031, 20261032, 20261033, 20261034, 20261035):
    RUNS.append((f"LP+MU seed {sd % 100}", f"a2[46]_medcmr_both_3b_seed{sd}_2*"))
for m, lab in (("lenprior", "LP"), ("mu", "MU")):
    for sd in (20261031, 20261032, 20261033):
        RUNS.append((f"{lab} seed {sd % 100}", f"a24_medcmr_{m}_3b_seed{sd}_2*"))
for sd in (20261011, 20261012, 20261013):
    RUNS.append((f"A20 tempered seed {sd % 100}", f"a20_medcmr_poeT_3b_seed{sd}_2*"))
for name, pat in RUNS:
    g = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}") if "smoke" not in x]
    if not g:
        continue
    P = {}
    for x in open(g[0] + "/rows_ft.jsonl"):
        r = json.loads(x)
        if r["group"] == "fresh" and r["case_id"] in T:
            P.setdefault(r["case_id"], {})[r["arm"]] = max(range(5), key=lambda k: r["direct"]["lp"][k])
    arms = ["base", "black", "mismatch_rand", "mismatch_nn"]
    gold = {c: "ABCDE".index(T[c]["answer"].strip()) for c in P if all(a in P[c] for a in arms)}
    out = []
    for lab, sel in (("all", list(gold)), ("cue-contradicting", [c for c in gold if gold[c] != shortest(c)]), ("cue-following", [c for c in gold if gold[c] == shortest(c)])):
        out.append(lab + f"(n={len(sel)}): " + " ".join(f"{a[:8]}={100 * sum(P[c][a] == gold[c] for c in sel) / len(sel):5.1f}" for a in arms))
    print(f"{name:22s} | " + " | ".join(out))
    n_all = len(gold)
    acc = lambda a: 100 * sum(P[c][a] == gold[c] for c in gold) / n_all
    print(f"   SUMMARY {name}: true {acc('base'):.2f} black {acc('black'):.2f} rand {acc('mismatch_rand'):.2f} nn {acc('mismatch_nn'):.2f} true-minus-nn {acc('base') - acc('mismatch_nn'):+.2f}")
