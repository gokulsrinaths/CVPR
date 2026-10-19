#!/usr/bin/env python
"""Shared pieces of the A6 length-rewrite experiment (docs/preregistration_v1_amendment_A6.md): prompt, label-free filters, case loading.
python lr_common.py selftest"""
import json
import os
import random
import re
import statistics as st
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
sys.path.insert(0, ROOT + "/scripts")
LETTERS = "ABCDE"
LAB = re.compile(r"^\s*[A-E]\)\s*")
NEG = {"no", "not", "without", "absent", "non", "never", "none", "neither", "nor", "lack", "lacks", "lacking", "negative"}
LAT = {"left", "right", "bilateral", "unilateral"}
HEDGE = {"generally", "typically", "likely", "may", "might", "possibly", "suggests", "suggesting", "suggest", "usually", "often", "could", "probable",
         "probably", "potentially", "indicating", "consistent", "appears", "apparent", "suggestive"}
SYS = "You are a careful medical editor."


def words(s):
    return len(s.split())


def nums(s):
    return sorted(re.findall(r"\d+(?:\.\d+)?", s))


def tokset(s, vocab):
    return frozenset(w for w in re.findall(r"[a-z']+", s.lower()) if w in vocab)


def hedges(s):
    return sum(1 for w in re.findall(r"[a-z]+", s.lower()) if w in HEDGE)


def target(orig):
    T = int(round(st.median([words(orig[l]) for l in LETTERS])))
    return T, int(round(0.9 * T)), int(round(1.1 * T))


def build_prompt(orig):
    T, lo, hi = target(orig)
    body = "\n".join(f"{l}: {orig[l]}" for l in LETTERS)
    return (f"Below are the five options of a multiple-choice question. Rewrite each option so that all five have about {T} words "
            f"(between {lo} and {hi} words), keeping each option's medical meaning exactly the same. Do not add new medical claims, do not remove key findings, "
            f"do not change numbers, laterality or negations, and do not add or remove hedging words (for example likely, may, suggests, typically). "
            f"Do not make any option more or less plausible than it was. Return only a JSON object with keys A, B, C, D, E.\n\n{body}")


def parse_options(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict) or set(d.keys()) != set(LETTERS):
        return None
    if not all(isinstance(d[l], str) and d[l].strip() for l in LETTERS):
        return None
    return {l: " ".join(d[l].split()) for l in LETTERS}


def passes(orig, new):
    """Label-free filters of A6. Returns (ok, reasons)."""
    why = []
    if new is None:
        return False, ["parse"]
    w = [words(new[l]) for l in LETTERS]
    if max(w) / max(1, min(w)) > 1.30:
        why.append("length_ratio")
    for l in LETTERS:
        if nums(orig[l]) != nums(new[l]):
            why.append("numbers_" + l)
        if tokset(orig[l], NEG) != tokset(new[l], NEG):
            why.append("negation_" + l)
        if tokset(orig[l], LAT) != tokset(new[l], LAT):
            why.append("laterality_" + l)
    return (not why), why


def load_cases():
    """{'test711': [MedReasonCase], 'dev700': [MedReasonCase]} (dev ids identical to H2's sample)."""
    import mvqa_diag as md
    from medreason_docker.schema import MedReasonCase, MedReasonOption
    T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}
    test = md.load_mcq_cases()
    dev = json.load(open(POOLD + "/split_v1/dev_ids.json"))
    dm = sorted(i for i in dev if T[i]["question type"] == "mcq")
    ds = sorted(random.Random(20261002).sample(dm, 700))
    dc = []
    for i in ds:
        opts = tuple(MedReasonOption(label=l, text=LAB.sub("", T[i][l]).strip()) for l in LETTERS)
        p = POOLD + "/extracted/train/imgs/" + os.path.basename(T[i]["image_path"])
        dc.append(MedReasonCase(case_id=i, task_type="mcq", question=T[i]["question"].strip(), image_paths=(__import__("pathlib").Path(p),), options=opts, metadata={}))
    return {"test711": test, "dev700": dc}, T


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n)
        if not ok:
            fails.append(n)
    orig = {"A": "No enhancement is seen in the left lobe, 5 mm lesion.", "B": "Diffuse enhancement is likely seen throughout the left lobe region today.",
            "C": "Mass is present on the right side.", "D": "Lesion is 12 mm and may be benign.", "E": "Nothing abnormal."}
    good = {"A": "No enhancement is seen in the left lobe, 5 mm lesion here.", "B": "Diffuse enhancement is likely seen across the whole left lobe region.",
            "C": "A mass is present on the right side of the organ here.", "D": "The lesion is 12 mm and may be benign on this scan.",
            "E": "Nothing abnormal is seen anywhere on the scan or image."}
    ok, why = passes(orig, good)
    check("a faithful, length-matched rewrite passes", ok and not why)
    bad = dict(good)
    bad["A"] = "Enhancement is seen in the left lobe, 5 mm lesion here."
    check("dropping a negation fails", not passes(orig, bad)[0] and "negation_A" in passes(orig, bad)[1])
    bad = dict(good)
    bad["D"] = "The lesion is 13 mm and may be benign overall."
    check("changing a number fails", "numbers_D" in passes(orig, bad)[1])
    bad = dict(good)
    bad["C"] = "A mass is present on the left side of the organ here."
    check("flipping laterality fails", "laterality_C" in passes(orig, bad)[1])
    bad = dict(good)
    bad["E"] = "Nothing."
    check("length ratio > 1.30 fails", "length_ratio" in passes(orig, bad)[1])
    check("parse of fenced JSON works", parse_options('```json\n{"A":"a","B":"b","C":"c","D":"d","E":"e"}\n```') == {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"})
    check("parse rejects missing key", parse_options('{"A":"a","B":"b"}') is None)
    check("parse rejects non-json", parse_options("sorry I cannot") is None)
    T_, lo, hi = target(orig)
    check("target is the rounded median word count with a +-10% band", T_ == int(round(st.median([words(orig[l]) for l in LETTERS]))) and lo <= T_ <= hi)
    p = build_prompt(orig)
    check("prompt carries no question and no label of the correct answer", "Question" not in p and "correct" not in p.lower().replace("do not make any option more or less plausible", ""))
    check("hedge counter", hedges("This is likely and may suggest benign") == 3)
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
