#!/usr/bin/env python
"""A38: three reproduction judges (Qwen2.5-VL-3B, -7B, -32B) against judge-side counterfactuals. For the 175 open-ended held-out cases and the outputs of the released adapter (A37 runs):
 C1 own trace, true image | C2 own trace, black image | C3 own trace, image of another case (same modality) | C4 own trace, no image | C5 trace and answer of another case (same modality), this case's true image and question
 C6 black-image-model trace, true image | C7 wrong-image-model trace, true image.
python a38_judge.py --judge 3b|7b|32b -> analysis/a38_scores_<judge>.json, analysis/a38_summary_<judge>.json"""
import argparse
import json
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--judge", choices=["3b", "7b", "32b"], required=True)
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
sys.path.insert(0, ROOT + "/scoring_suite/scoring")
from medreason_scoring.judges import VA_RUBRIC, clamp04  # noqa: E402
from medreason_scoring.metrics import va_final  # noqa: E402
from PIL import Image  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

SRC = ROOT + "/data/holdout886"
cases = {c["case_id"]: c for c in json.load(open(SRC + "/cases.json"))["cases"] if c["task_type"] == "open"}
PRED = {a: {p["case_id"]: p for p in json.load(open(f"{ROOT}/data/holdout886_open_{a}/run_output/results.json"))["answers"]} for a in ("true", "black", "wrong")}
OTHER = json.load(open(ROOT + "/data/holdout886_open_wrong/assignment.json"))["assignment"]
path = f"{ROOT}/models/qwen25vl_{args.judge}_instruct"
proc = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto" if args.judge == "32b" else "cuda")
model.eval()


def judge(question, pred, image):
    prompt = VA_RUBRIC.format(question=question, answer=pred.get("answer", ""), trace=pred.get("reasoning_trace", ""))
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[image] if image is not None else None, return_tensors="pt")
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


def img(cid):
    return Image.open(f"{SRC}/{cases[cid]['image_path']}").convert("RGB")


ids = sorted(cases)
SC = {k: {} for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}
for n, cid in enumerate(ids):
    q = cases[cid]["question"]
    real = img(cid)
    o = OTHER[cid]
    SC["C1"][cid] = judge(q, PRED["true"][cid], real)
    SC["C2"][cid] = judge(q, PRED["true"][cid], Image.new("RGB", real.size, (0, 0, 0)))
    SC["C3"][cid] = judge(q, PRED["true"][cid], img(o))
    SC["C4"][cid] = judge(q, PRED["true"][cid], None)
    SC["C5"][cid] = judge(q, PRED["true"][o], real)
    SC["C6"][cid] = judge(q, PRED["black"][cid], real)
    SC["C7"][cid] = judge(q, PRED["wrong"][cid], real)
    if (n + 1) % 25 == 0:
        print(f"[a38 {args.judge}] {n + 1}/{len(ids)}", flush=True)
json.dump(SC, open(f"{ROOT}/analysis/a38_scores_{args.judge}.json", "w"))
rng = np.random.RandomState(20261108)


def paired(a, b, m="RVF_trace"):
    x = np.array([SC[a][i][m] for i in ids], float)
    y = np.array([SC[b][i][m] for i in ids], float)
    d = x - y
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"mean_a": round(float(x.mean()), 3), "mean_b": round(float(y.mean()), 3), "diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}


out = {"judge": f"Qwen2.5-VL-{args.judge.upper()}-Instruct (reproduction; rubric of the scoring suite)", "n": len(ids),
       "means": {k: {m: round(float(np.mean([v[i][m] for i in ids])), 3) for m in ("VA_answer", "RVF_trace", "VA_final")} for k, v in SC.items()},
       "K1_image_identity_C1_minus_C3": paired("C1", "C3"), "image_presence_C1_minus_C2": paired("C1", "C2"), "image_vs_none_C1_minus_C4": paired("C1", "C4"),
       "K2_mismatched_trace_C1_minus_C5": paired("C1", "C5"), "K3_black_model_C1_minus_C6": paired("C1", "C6"), "wrong_model_C1_minus_C7": paired("C1", "C7")}
out["K1"] = bool(out["K1_image_identity_C1_minus_C3"]["diff"] >= 0.5 and out["K1_image_identity_C1_minus_C3"]["ci95"][0] >= 0.25)
out["K2"] = bool(out["K2_mismatched_trace_C1_minus_C5"]["diff"] >= 1.0 and out["K2_mismatched_trace_C1_minus_C5"]["ci95"][0] >= 0.5)
out["K3"] = bool(out["K3_black_model_C1_minus_C6"]["diff"] >= 1.0 and out["K3_black_model_C1_minus_C6"]["ci95"][0] >= 0.5)
json.dump(out, open(f"{ROOT}/analysis/a38_summary_{args.judge}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
