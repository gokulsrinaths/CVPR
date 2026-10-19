#!/usr/bin/env python
"""A37 judge and analysis: scores the released adapter's open-ended outputs (true / black / wrong image given to the model) with a reproduction of the organisers' VA judge (Qwen2.5-VL-7B-Instruct,
rubric of scoring_suite/medreason_scoring/judges.py; the judge sees the TRUE image of the case, or a black image for the judge-side control) and evaluates the frozen rules of
docs/preregistration_v1_amendment_A37.md. python a37_judge.py -> analysis/a37_scores.json, analysis/a37_summary.json"""
import json
import re
import sys

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
sys.path.insert(0, ROOT + "/scoring_suite/scoring")
from medreason_scoring.judges import VA_JUDGE_MODEL, VA_RUBRIC, clamp04  # noqa: E402
from medreason_scoring.metrics import va_final  # noqa: E402
from PIL import Image  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

SRC = ROOT + "/data/holdout886"
cases = {c["case_id"]: c for c in json.load(open(SRC + "/cases.json"))["cases"] if c["task_type"] == "open"}
PRED = {}
for arm in ("true", "black", "wrong"):
    r = json.load(open(f"{ROOT}/data/holdout886_open_{arm}/run_output/results.json"))["answers"]
    PRED[arm] = {p["case_id"]: p for p in r}
    assert set(PRED[arm]) == set(cases), arm
proc = AutoProcessor.from_pretrained(ROOT + "/models/qwen25vl_7b_instruct")
model = AutoModelForImageTextToText.from_pretrained(ROOT + "/models/qwen25vl_7b_instruct", dtype=torch.bfloat16, device_map="cuda")
model.eval()


def judge(cid, pred, image_mode):
    img = Image.open(f"{SRC}/{cases[cid]['image_path']}").convert("RGB")
    if image_mode == "black":
        img = Image.new("RGB", img.size, (0, 0, 0))
    prompt = VA_RUBRIC.format(question=cases[cid]["question"], answer=pred.get("answer", ""), trace=pred.get("reasoning_trace", ""))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = proc.batch_decode(out[:, inp["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    va = rvf = 0
    try:
        obj = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        va, rvf = obj.get("VA_answer", 0), obj.get("RVF_trace", 0)
    except Exception:
        nums = re.findall(r"[0-4]", raw)
        if nums:
            va = nums[0]
            rvf = nums[1] if len(nums) > 1 else nums[0]
    va, rvf = clamp04(va), clamp04(rvf)
    return {"VA_answer": va, "RVF_trace": rvf, "VA_final": va_final(va, rvf), "raw": raw.strip()}


SCORES = {}
for key, (parm, jmode) in {"true_model__judge_true": ("true", "true"), "black_model__judge_true": ("black", "true"), "wrong_model__judge_true": ("wrong", "true"), "true_model__judge_black": ("true", "black")}.items():
    SCORES[key] = {}
    for n, cid in enumerate(sorted(cases)):
        SCORES[key][cid] = judge(cid, PRED[parm][cid], jmode)
    print(key, "done", flush=True)
json.dump(SCORES, open(ROOT + "/analysis/a37_scores.json", "w"))

rng = np.random.RandomState(20261107)
ids = sorted(cases)


def paired(a_key, b_key, metric):
    a = np.array([SCORES[a_key][i][metric] for i in ids], float)
    b = np.array([SCORES[b_key][i][metric] for i in ids], float)
    d = a - b
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"mean_a": round(float(a.mean()), 3), "mean_b": round(float(b.mean()), 3), "diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}


out = {"judge": VA_JUDGE_MODEL + " (reproduction; the organisers' prompts and model size are not known)", "n": len(ids), "means": {k: {m: round(float(np.mean([v[i][m] for i in ids])), 3) for m in ("VA_answer", "RVF_trace", "VA_final")} for k, v in SCORES.items()}}
out["M1_RVF_true_minus_black_model"] = paired("true_model__judge_true", "black_model__judge_true", "RVF_trace")
out["M2_VAfinal_true_minus_black_model"] = paired("true_model__judge_true", "black_model__judge_true", "VA_final")
out["M3_RVF_true_minus_wrong_model"] = paired("true_model__judge_true", "wrong_model__judge_true", "RVF_trace")
out["J1_RVF_judge_true_minus_judge_black"] = paired("true_model__judge_true", "true_model__judge_black", "RVF_trace")
out["M1"] = bool(out["M1_RVF_true_minus_black_model"]["diff"] >= 1.0 and out["M1_RVF_true_minus_black_model"]["ci95"][0] >= 0.5)
out["M2"] = bool(out["M2_VAfinal_true_minus_black_model"]["diff"] >= 0.5 and out["M2_VAfinal_true_minus_black_model"]["ci95"][0] >= 0.25)
out["M3"] = bool(out["M3_RVF_true_minus_wrong_model"]["diff"] >= 1.0 and out["M3_RVF_true_minus_wrong_model"]["ci95"][0] >= 0.5)
out["J1"] = bool(out["J1_RVF_judge_true_minus_judge_black"]["diff"] >= 0.5 and out["J1_RVF_judge_true_minus_judge_black"]["ci95"][0] >= 0.25)
blind = re.compile(r"\b(black|blank|no image|cannot see|can't see|not visible|unable to see|no visible)\b", re.I)
out["share_traces_mentioning_black_or_blank"] = {a: round(float(np.mean([bool(blind.search(PRED[a][i].get("reasoning_trace", ""))) for i in ids])), 3) for a in PRED}
out["mean_trace_chars"] = {a: round(float(np.mean([len(PRED[a][i].get("reasoning_trace", "")) for i in ids])), 1) for a in PRED}
out["answer_identical_to_true_arm"] = {a: round(float(np.mean([PRED[a][i].get("answer", "").strip().lower() == PRED["true"][i].get("answer", "").strip().lower() for i in ids])), 3) for a in ("black", "wrong")}
json.dump(out, open(ROOT + "/analysis/a37_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
