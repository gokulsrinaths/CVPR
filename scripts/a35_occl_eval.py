#!/usr/bin/env python
"""A35: where in the image does the evidence sit? Evaluation only. Loads a saved dose-response adapter (A17 PMC-VQA or A21 A-OKVQA, Qwen2.5-VL-3B) and evaluates E1 with planted options and the arms
true image, black image, centre occluded (central rectangle of half the image area set to black) and border occluded (the complementary frame of half the area set to black).
python a35_occl_eval.py --bench pmc|aokvqa --adapter RUN_DIR --dose D --label L"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bench", choices=["pmc", "aokvqa"], required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--dose", type=float, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SPL = ROOT + ("/data/pmcvqa/split_a11" if args.bench == "pmc" else "/data/aokvqa/split_a21")
MODEL = ROOT + "/models/qwen25vl_3b_instruct"
MAXSIDE = 448
LETTERS = "ABCD"
SYS = "You are a medical imaging expert." if args.bench == "pmc" else "You are a helpful assistant that answers multiple-choice questions about images."
INSTR = "Answer with only the letter (A, B, C or D) of the correct option."
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dose_plant import assign_u, length_options  # noqa: E402

REC = json.load(open(SPL + "/records.json"))
IDS = json.load(open(SPL + "/ids.json"))
R = {}
for g in REC.values():
    R.update(g)
U = assign_u(IDS["E1"])
PR = {cid: {"choices": length_options(cid, R[cid]["choices"], LETTERS.index(R[cid]["gold"]), U[cid], args.dose), "gold": R[cid]["gold"]} for cid in IDS["E1"]}
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
run_dir = Path(ROOT) / "experiments" / "pilot" / f"a35_occl_{args.label}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
json.dump({"amendment": "A35", "bench": args.bench, "adapter": args.adapter, "dose": args.dose, "label": args.label, "smoke": args.smoke}, open(run_dir / "config.json", "w"), indent=1)
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model = PeftModel.from_pretrained(model, args.adapter + "/adapter" if os.path.isdir(args.adapter + "/adapter") else args.adapter)
model.eval()
dev = model.device
LID = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]


def load_img(cid):
    im = Image.open(R[cid]["image"]).convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s < 1 else im


def occlude(im, mode):
    a = np.array(im)
    h, w = a.shape[:2]
    ch, cw = int(round(h * math.sqrt(0.5))), int(round(w * math.sqrt(0.5)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    if mode == "occl_center":
        a[y0:y0 + ch, x0:x0 + cw] = 0
    else:
        keep = a[y0:y0 + ch, x0:x0 + cw].copy()
        a[:] = 0
        a[y0:y0 + ch, x0:x0 + cw] = keep
    return Image.fromarray(a)


def user_text(cid):
    return R[cid]["question"] + "\n\nOptions:\n" + "\n".join(f"  {l}: {c}" for l, c in zip(LETTERS, PR[cid]["choices"])) + "\n\n" + INSTR


@torch.inference_mode()
def lp_letters(cid, img):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(cid)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
    lp = torch.log_softmax(model(**inp).logits[0, -1].float(), -1)
    return [float(lp[i]) for i in LID]


ids = IDS["E1"][:20] if args.smoke else IDS["E1"]
fh = open(run_dir / "rows_occl.jsonl", "w")
t0 = time.time()
for n, cid in enumerate(ids):
    real = load_img(cid)
    for arm in ("base", "black", "occl_center", "occl_border"):
        img = real if arm == "base" else Image.fromarray(np.zeros((224, 224, 3), np.uint8)) if arm == "black" else occlude(real, arm)
        fh.write(json.dumps({"case_id": cid, "arm": arm, "gold": PR[cid]["gold"], "lp": lp_letters(cid, img)}) + "\n")
    if (n + 1) % 200 == 0:
        fh.flush()
        print(f"[occl] {n + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
fh.close()
print("RUN_DIR", run_dir, flush=True)
os._exit(0)
