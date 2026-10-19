"""Cue audit: how often is the correct option the shortest / longest / at a fixed position?
Text-only statistics; no model, no image. Exploratory (not pre-registered)."""
import ast, glob, json, os, re, sys, zipfile
from collections import Counter
import csv
import pyarrow.parquet as pq


def rows(f):
    return pq.read_table(f).to_pylist()

D = "data"
OUT = []


def strip_letter(s):
    return re.sub(r"^\s*\(?[A-Za-z]\)?[\.\):]\s*", "", str(s)).strip()


def stats(name, items, note=""):
    """items: list of (options:list[str], gold_index:int)"""
    n = 0
    k_c = Counter()
    sh = lo = 0.0  # expected-credit counts (ties split)
    sh_strict = lo_strict = 0
    ties = 0
    pos = Counter()
    chance = 0.0
    for opts, g in items:
        if not opts or g is None or not (0 <= g < len(opts)):
            continue
        L = [len(o) for o in opts]
        n += 1
        k_c[len(opts)] += 1
        chance += 1.0 / len(opts)
        mn, mx = min(L), max(L)
        nm = L.count(mn)
        nx = L.count(mx)
        if L[g] == mn:
            sh += 1.0 / nm
            if nm == 1:
                sh_strict += 1
        if L[g] == mx:
            lo += 1.0 / nx
            if nx == 1:
                lo_strict += 1
        if nm > 1:
            ties += 1
        pos[g] += 1
    if n == 0:
        return
    maxpos, maxshare = max(pos.items(), key=lambda kv: kv[1])
    r = dict(
        benchmark=name, n=n, options=dict(k_c), chance=round(100 * chance / n, 1),
        shortest_rule_acc=round(100 * sh / n, 1), shortest_strict=round(100 * sh_strict / n, 1),
        longest_rule_acc=round(100 * lo / n, 1), longest_strict=round(100 * lo_strict / n, 1),
        tie_share=round(100 * ties / n, 1), top_position=int(maxpos),
        top_position_share=round(100 * maxshare / n, 1), note=note)
    OUT.append(r)
    print(json.dumps(r), flush=True)


# --- Med-CMR (MCQ cases of the official training selection)
try:
    cs = json.load(open(f"{D}/train_pool/official_json/medreason_train_selection.json"))["cases"]
    items = []
    for c in cs:
        if c.get("question type") != "mcq":
            continue
        letters = [x for x in "ABCDEFGH" if x in c]
        opts = [strip_letter(c[x]) for x in letters]
        items.append((opts, letters.index(c["answer"])) if c["answer"] in letters else (None, None))
    stats("Med-CMR (train MCQ)", items, "challenge training pool")
except Exception as e:
    print("medcmr fail", e)

# --- PMC-VQA
for f, nm in (("train.csv", "PMC-VQA train"), ("test_clean.csv", "PMC-VQA test_clean")):
    try:
        items = []
        for r in csv.DictReader(open(f"{D}/pmcvqa/{f}", encoding="utf-8", newline="")):
            opts = [strip_letter(r[f"Choice {x}"]) for x in "ABCD" if r.get(f"Choice {x}")]
            lab = str(r["Answer_label"]).strip()
            if lab not in "ABCD" or not lab or len(opts) < 4:
                continue
            g = "ABCD".index(lab)
            items.append((opts, g))
        stats(nm, items)
    except Exception as e:
        print(nm, "fail", e)

# --- OmniMedVQA (open-access subset, full)
try:
    z = zipfile.ZipFile(f"{D}/omnimed/OmniMedVQA.zip")
    items = []
    for n in z.namelist():
        if "QA_information/Open-access/" in n and n.endswith(".json"):
            for q in json.loads(z.read(n)):
                opts = [str(q.get(f"option_{x}")) for x in "ABCD" if q.get(f"option_{x}") is not None]
                ans = q.get("gt_answer")
                if ans in opts:
                    items.append((opts, opts.index(ans)))
    stats("OmniMedVQA (open-access)", items)
except Exception as e:
    print("omni fail", e)

# --- A-OKVQA (validation parquet)
try:
    fs = glob.glob(f"{D}/aokvqa/raw/data/validation-*.parquet")
    items = [(list(r["choices"]), int(r["correct_choice_idx"])) for f in fs for r in rows(f)]
    stats("A-OKVQA (val)", items, "photographs")
except Exception as e:
    print("aokvqa fail", e)

# --- MedXpertQA-MM
try:
    items = []
    for ln in open(f"{D}/cue_audit/medxpert_test.jsonl"):
        q = json.loads(ln)
        ks = sorted(q["options"])
        items.append(([q["options"][k] for k in ks], ks.index(q["label"])))
    stats("MedXpertQA-MM (test)", items, "expert board-style")
except Exception as e:
    print("medxpert fail", e)

# --- MMMU health subjects (validation, MCQ only)
try:
    items = []
    for f in glob.glob(f"{D}/cue_audit/mmmu_*.parquet"):
        for r in rows(f):
            if r["question_type"] != "multiple-choice":
                continue
            opts = ast.literal_eval(r["options"])
            items.append((opts, "ABCDEFGHI".index(r["answer"])))
    stats("MMMU health (val, 5 subjects)", items)
except Exception as e:
    print("mmmu fail", e)

json.dump(OUT, open("analysis/cue_audit_results.json", "w"), indent=1)
