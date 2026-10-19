#!/usr/bin/env python
"""A27 analysis: exactly rule C and the reading rule of docs/preregistration_v1_amendment_A27.md. python a27_analysis.py -> analysis/a27_summary.json"""
import glob
import json

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
MODELS = {"ul41": "a26_medcmr_ul_3b_seed20261041_2*", "ul42": "a26_medcmr_ul_3b_seed20261042_2*", "ul43": "a26_medcmr_ul_3b_seed20261043_2*",
          "lpmu31": "a24_medcmr_both_3b_seed20261031_2*", "lpmu32": "a24_medcmr_both_3b_seed20261032_2*", "a9": "ft_3b_s0_22569048_*"}
LEVELS = ["blur32", "blur16", "blur8", "blur4", "blur2", "base"]


def accs(path, group, arms):
    P = {}
    for x in open(path):
        r = json.loads(x)
        if r["group"] == group and r["arm"] in arms:
            P.setdefault(r["arm"], {})[r["case_id"]] = (max(range(5), key=lambda k: r["direct"]["lp"][k]), r["gold"])
    return {a: 100 * sum(p == "ABCDE".index(g) for p, g in P[a].values()) / len(P[a]) for a in P}


out = {"models": {}}
for lab, pat in MODELS.items():
    b = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a27_blur_{lab}_2*/rows_blur.jsonl")]
    r0 = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}/rows_ft.jsonl") if "smoke" not in x]
    if not b or not r0:
        out["models"][lab] = {"incomplete": True}
        continue
    A = accs(sorted(b)[-1], "fresh", LEVELS)
    black = accs(r0[0], "fresh", ["black"])["black"]
    seq = [A[l] for l in LEVELS]
    rise = seq[-1] - seq[0]
    steps_ok = all(seq[i + 1] - seq[i] >= -1.0 for i in range(len(seq) - 1))
    gap = seq[-1] - black
    out["models"][lab] = {"acc": {l: round(A[l], 2) for l in LEVELS}, "black": round(black, 2), "rise_blur32_to_intact": round(rise, 2), "C": bool(rise >= 10.0 and steps_ok), "R8": round((A["blur8"] - black) / gap, 3) if abs(gap) > 1e-9 else None,
                          "R2": round((A["blur2"] - black) / gap, 3) if abs(gap) > 1e-9 else None}
ul = [out["models"].get(k, {}) for k in ("ul41", "ul42", "ul43")]
if all("C" in m for m in ul):
    out["UL_content_dependent"] = bool(sum(m["C"] for m in ul) >= 2)
    out["UL_R8_ge_0.8_seeds"] = int(sum((m["R8"] is not None and m["R8"] >= 0.8) for m in ul))
json.dump(out, open(ROOT + "/analysis/a27_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
