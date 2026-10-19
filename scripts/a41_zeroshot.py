#!/usr/bin/env python
"""A41: zero-shot Qwen2.5-VL (7B or 32B, no fine-tuning) on six multiple-choice benchmarks with the true image, a black image and no image plus an explicit note that the image is unavailable
(a MIRAGE-style test on open models). Letter log-probability readout. python a40_zeroshot.py --model 7b|32b [--smoke] -> experiments/pilot/a40_<model>_*/rows.jsonl"""
import argparse
import ast
import glob
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["3b"], required=True)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
D = ROOT + "/data"
LETTERS = "ABCDEFGHIJ"
MAXSIDE = 448
SYS = "You are a helpful assistant that answers multiple-choice questions."


def from_split(path, group):
    rec = {}
    for g in json.load(open(path + "/records.json")).values():
        rec.update(g)
    ids = json.load(open(path + "/ids.json"))[group]
    return [{"id": i, "question": rec[i]["question"], "options": rec[i]["choices"], "gold": "ABCD".index(rec[i]["gold"]), "images": [rec[i]["image"]]} for i in ids]


def medcmr():
    cases = {c["case_id"]: c for c in json.load(open(D + "/holdout886/cases.json"))["cases"] if c["task_type"] == "mcq"}
    gt = {g["case_id"]: g["answer"].strip() for g in json.load(open(D + "/holdout886/ground_truth.json"))["ground_truth"]}
    out = []
    for i, c in sorted(cases.items()):
        opts = [o["text"] for o in sorted(c["options"], key=lambda o: o["label"])]
        out.append({"id": i, "question": c["question"], "options": opts, "gold": "ABCDE".index(gt[i]), "images": [D + "/holdout886/" + c["image_path"]]})
    return out


def medx():
    z = zipfile.ZipFile(D + "/cue_audit/medxpert_images.zip")
    by = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
    out = []
    for ln in open(D + "/cue_audit/medxpert_test.jsonl"):
        q = json.loads(ln)
        ks = sorted(q["options"])
        imgs = [("zip", z, by[x]) for x in q["images"] if x in by]
        if imgs:
            out.append({"id": q["id"], "question": q["question"].split("\nAnswer Choices:")[0], "options": [q["options"][k] for k in ks], "gold": ks.index(q["label"]), "images": imgs})
    return out


def mmmu():
    out = []
    for f in sorted(glob.glob(D + "/cue_audit/mmmu_*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if r["question_type"] != "multiple-choice":
                continue
            imgs = [("bytes", r[f"image_{k}"]["bytes"]) for k in range(1, 4) if r.get(f"image_{k}")]
            if imgs:
                out.append({"id": r["id"], "question": r["question"], "options": ast.literal_eval(r["options"]), "gold": "ABCDEFGHI".index(r["answer"]), "images": imgs})
    return out


BENCH = {"Med-CMR": medcmr(), "PMC-VQA": from_split(D + "/pmcvqa/split_a11", "E1"), "OmniMedVQA": from_split(D + "/omnimed/split_a16", "E1"), "A-OKVQA": from_split(D + "/aokvqa/split_a21", "E1"),
         "MedXpertQA-MM": medx(), "MMMU health": mmmu()}
if args.smoke:
    BENCH = {k: v[:12] for k, v in BENCH.items()}
print({k: len(v) for k, v in BENCH.items()}, flush=True)


def load(img):
    if isinstance(img, str):
        im = Image.open(img)
    elif img[0] == "zip":
        im = Image.open(io.BytesIO(img[1].read(img[2])))
    else:
        im = Image.open(io.BytesIO(img[1]))
    im = im.convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s < 1 else im


import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

path = f"{ROOT}/models/qwen25vl_{args.model}_instruct"
proc = AutoProcessor.from_pretrained(path)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto" if args.model == "32b" else "cuda", attn_implementation="sdpa")
model.eval()
dev = model.device
LID = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
assert all(len(x) == 1 for x in LID), LID
LID = [x[0] for x in LID]
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
run_dir = Path(ROOT) / "experiments" / "pilot" / f"a41_{args.model}{'_smoke' if args.smoke else ''}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
json.dump({"amendment": "A41", "model": path, "smoke": args.smoke, "sizes": {k: len(v) for k, v in BENCH.items()}, "script_sha256": __import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest()}, open(run_dir / "config.json", "w"), indent=1)
NOTE = "Note: the image could not be loaded. Answer using the question and options only.\n\n"


@torch.inference_mode()
def lp(item, arm):
    n = len(item["options"])
    body = item["question"] + "\n\nOptions:\n" + "\n".join(f"  {LETTERS[k]}: {o}" for k, o in enumerate(item["options"])) + f"\n\nAnswer with only the letter ({', '.join(LETTERS[:n])}) of the correct option."
    if arm == "told":
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "text", "text": NOTE + body}]}]
        images = None
    else:
        real = [load(x) for x in item["images"]]
        images = real if arm == "true" else [Image.new("RGB", im.size, (0, 0, 0)) for im in real]
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": body}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=images, return_tensors="pt").to(dev)
    l = torch.log_softmax(model(**inp).logits[0, -1].float(), -1)
    return [float(l[LID[k]]) for k in range(n)]


fh = open(run_dir / "rows.jsonl", "w")
t0, done, total = time.time(), 0, 3 * sum(len(v) for v in BENCH.values())
for name, items in BENCH.items():
    for it in items:
        for arm in ("true", "black", "told"):
            fh.write(json.dumps({"bench": name, "id": it["id"], "arm": arm, "gold": it["gold"], "lp": lp(it, arm)}) + "\n")
            done += 1
            if done % 500 == 0:
                fh.flush()
                print(f"[a40 {args.model}] {done}/{total} {(time.time() - t0) / 60:.1f} min", flush=True)
fh.close()
print("RUN_DIR", run_dir, flush=True)
os._exit(0)
