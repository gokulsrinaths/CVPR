#!/usr/bin/env python
"""A7 helpers: per-option acceptance rule, bands, prompts. python lr_common2.py selftest"""
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_common import HEDGE, LAT, LETTERS, NEG, SYS, nums, tokset, words  # noqa: E402,F401


def bands(orig):
    Tc = int(st.median([len(orig[l]) for l in LETTERS]))
    Tw = int(st.median([words(orig[l]) for l in LETTERS]))
    return {"Tc": Tc, "Tw": Tw, "c_lo": min(math.floor(0.92 * Tc), Tc - 4), "c_hi": max(math.ceil(1.08 * Tc), Tc + 4),
            "w_lo": max(1, math.floor(0.85 * Tw)), "w_hi": max(math.ceil(1.15 * Tw), Tw + 1)}


def clean(text):
    t = text.strip().strip('"').strip()
    t = re.sub(r"^(rewritten option|option|rewrite)\s*:\s*", "", t, flags=re.I).strip().strip('"').strip()
    return t


def option_ok(orig, new, b):
    why = []
    if "\n" in new.strip():
        why.append("multiline")
    new = " ".join(new.split())
    c, w = len(new), words(new)
    if not (b["c_lo"] <= c <= b["c_hi"]):
        why.append("chars")
    if not (b["w_lo"] <= w <= b["w_hi"]):
        why.append("words")
    if nums(orig) != nums(new):
        why.append("numbers")
    if tokset(orig, NEG) != tokset(new, NEG):
        why.append("negation")
    if tokset(orig, LAT) != tokset(new, LAT):
        why.append("laterality")
    if tokset(orig, HEDGE) != tokset(new, HEDGE):
        why.append("hedge")
    return (not why), why


def len_err(new, b):
    c = len(" ".join(new.split()))
    return max(0, b["c_lo"] - c, c - b["c_hi"])


def first_prompt(text, b):
    return (f"Rewrite the following answer option from a medical multiple-choice question in different words so that it has between {b['c_lo']} and {b['c_hi']} characters "
            f"(about {b['w_lo']} to {b['w_hi']} words), keeping its medical meaning exactly the same. Do not add new medical claims, do not remove key findings, do not change numbers, "
            f"laterality or negations, and do not add or remove hedging words (for example likely, may, suggests, typically). Return only the rewritten option text.\n\nOption: {text}")


def feedback(new, b):
    new = " ".join(new.split())
    return (f"That rewrite has {len(new)} characters and {words(new)} words. It must have between {b['c_lo']} and {b['c_hi']} characters and between {b['w_lo']} and {b['w_hi']} words, "
            f"with the same meaning. Return only the corrected option text.")


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n)
        if not ok:
            fails.append(n)
    orig = {"A": "No enhancement is seen in the left lobe.", "B": "Diffuse enhancement is likely seen across the whole left lobe region.",
            "C": "A mass is present on the right side.", "D": "The lesion is 12 mm and may be benign.", "E": "Nothing abnormal."}
    b = bands(orig)
    print("     bands:", b)
    check("Tc and Tw are medians of the originals", b["Tc"] == int(st.median([len(orig[l]) for l in LETTERS])) and b["Tw"] == int(st.median([words(orig[l]) for l in LETTERS])))
    check("band always contains the target and has min +-4 chars", b["c_lo"] <= b["Tc"] - 4 and b["c_hi"] >= b["Tc"] + 4)
    cand = "No enhancement is noted in the left lobe."
    okA, whyA = option_ok(orig["A"], cand, b)
    print("     faithful candidate:", len(cand), "chars", words(cand), "words ->", okA, whyA)
    check("a faithful in-band rewrite is accepted", okA and not whyA)
    check("dropping a negation is rejected", "negation" in option_ok(orig["A"], "Enhancement is noted in the left lobe here ok.", b)[1])
    check("changing a number is rejected", "numbers" in option_ok(orig["D"], "The lesion is 13 mm and may be benign.", b)[1])
    check("flipping laterality is rejected", "laterality" in option_ok(orig["C"], "A mass is present on the left side.", b)[1])
    check("adding a hedge is rejected", "hedge" in option_ok(orig["C"], "A mass is likely present on the right side.", b)[1])
    check("removing a hedge is rejected", "hedge" in option_ok(orig["D"], "The lesion is 12 mm and is benign here.", b)[1])
    check("too short in characters is rejected", "chars" in option_ok(orig["E"], "Normal.", b)[1])
    check("too long in characters is rejected", "chars" in option_ok(orig["B"], orig["B"], b)[1])
    check("multi-line is rejected", "multiline" in option_ok(orig["A"], "line one" + chr(10) + "line two", b)[1])
    check("clean strips prefix and quotes", clean('Option: "A mass is present."') == "A mass is present.")
    check("len_err is 0 inside the band and positive outside", len_err("x" * b["Tc"], b) == 0 and len_err("x", b) > 0)
    p = first_prompt(orig["A"], b)
    check("prompt has no question and no other option text", "Diffuse" not in p and "Question" not in p)
    check("feedback reports counts", "characters and" in feedback("abc def", b))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
