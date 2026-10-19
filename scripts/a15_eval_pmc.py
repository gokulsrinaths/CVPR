#!/usr/bin/env python
"""A15: adapter-only evaluation with extra image arms (wrong images, blurred images) on PMC-VQA E1. Derived from pmc_run2.py (A13). Original: A13: plant an option-length cue in PMC-VQA (filler on all options; correct option shortest with p = 0.92), fine-tune with the A9/A11 recipe on the planted training set, evaluate on planted and original questions.
python pmc_run2.py --model q3b|q7b|ivl2b [--smoke] | --selftest --model q3b   Never overwrites a run."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["q3b", "q7b", "ivl2b"], required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--adapter", required=False)
ap.add_argument("--cond", choices=["orig", "implant"], default="orig")
ap.add_argument("--arms", default="base,black,mismatch_rand,mismatch_nn")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SPL = ROOT + "/data/pmcvqa/split_a11"
MODEL = {"q3b": "qwen25vl_3b_instruct", "q7b": "qwen25vl_7b_instruct", "ivl2b": "internvl3_2b_hf"}[args.model]
MODEL = f"{ROOT}/models/{MODEL}"
SEED, ACC, LR, MAXSIDE = 20261003, 16, 1e-4, 448
LETTERS = "ABCD"
SYS = "You are a medical imaging expert."
INSTR = "Answer with only the letter (A, B, C or D) of the correct option."
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_implant import implant_options  # noqa: E402

REC = json.load(open(SPL + "/records.json"))
IDS = json.load(open(SPL + "/ids.json"))
DON = json.load(open(SPL + "/donors.json"))
R = {}
for g in REC.values():
    R.update(g)
GRP = {i: g for g, v in IDS.items() for i in v}


def load_img(cid):
    im = Image.open(R[cid]["image"]).convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s < 1 else im


def user_text(cid, cond="orig"):
    r = R[cid]
    ch = implant_options(cid, r["choices"], LETTERS.index(r["gold"]))[0] if cond == "implant" else r["choices"]
    return r["question"] + "\n\nOptions:\n" + "\n".join(f"  {l}: {c}" for l, c in zip(LETTERS, ch)) + "\n\n" + INSTR


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    check("sizes", [len(IDS[g]) for g in ("E1", "E2", "VAL700", "TR")] == [1999, 3000, 700, 9277])
    check("TR disjoint from E1, E2, VAL700 (ids and articles)", not (set(IDS["TR"]) & (set(IDS["E1"]) | set(IDS["E2"]) | set(IDS["VAL700"]))) and not ({R[i]["article"] for i in IDS["TR"]} & {R[i]["article"] for g in ("E1", "E2", "VAL700") for i in IDS[g]}))
    check("user text format", user_text(IDS["TR"][0]).endswith(INSTR) and "Options:\n  A: " in user_text(IDS["TR"][0]))
    check("image downscaled", max(load_img(IDS["TR"][0]).size) <= MAXSIDE)
    check("model dir", os.path.exists(MODEL + "/config.json"))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


if args.selftest:
    selftest()

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

jobid = os.environ.get("SLURM_JOB_ID", "nojob")
tag = f"a15pmc_{args.model}_{args.cond}{'_smoke' if args.smoke else ''}"
run_dir = Path(ROOT) / "experiments" / "pilot" / f"{tag}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"amendment": "A15", "adapter_in": args.adapter, "cond": args.cond, "arms": args.arms, "model": MODEL, "smoke": args.smoke, "script_sha256": sha(__file__), "amendment_sha256": sha(ROOT + "/docs/preregistration_v1_amendment_A15.md"),
           "split_sha256": json.load(open(SPL + "/SHA256.json")), "recipe": {"lora_r": 16, "alpha": 32, "dropout": 0.05, "lr": LR, "accum": ACC, "epochs": 1, "seed": SEED, "max_side": MAXSIDE},
           "sizes": {g: len(v) for g, v in IDS.items()}}, open(run_dir / "config.json", "w"), indent=2)
proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
dev = model.device
LID = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
assert all(len(x) == 1 for x in LID), LID
LID = [x[0] for x in LID]
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
assert isinstance(IM_END, int) and IM_END >= 0
json.dump({"host": os.uname().nodename, "slurm_job_id": jobid, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "letter_ids": LID, "im_end": IM_END}, open(run_dir / "env.json", "w"), indent=2)


def prompt_inputs(cid, img, cond="orig"):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(cid, cond)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=[text], images=[img], return_tensors="pt")


@torch.inference_mode()
def lp_letters(cid, img, cond="orig"):
    inp = prompt_inputs(cid, img, cond).to(dev)
    lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    letters = [float(lp[i]) for i in LID]
    return {"lp": letters, "mass": float(sum(math.exp(x) for x in letters))}


def evaluate(stage, plan, limit=0, cond="orig"):
    model.eval()
    fh = open(run_dir / f"rows_{stage}.jsonl", "a")
    n, t0 = 0, time.time()
    total = sum(len(ids[:limit] if limit else ids) * len(arms) for _, ids, arms in plan)
    for g, ids, arms in plan:
        for cid in (ids[:limit] if limit else ids):
            real = load_img(cid)
            for a in arms:
                if a == "base":
                    img = real
                elif a == "black":
                    img = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                elif a in ("blur8", "blur24"):
                    img = real.filter(ImageFilter.GaussianBlur(int(a[4:])))
                else:
                    img = load_img(DON[g]["rand" if a == "mismatch_rand" else "nn"][cid])
                fh.write(json.dumps({"stage": stage, "cond": cond, "group": g, "case_id": cid, "arm": a, "gold": R[cid]["gold"], "direct": lp_letters(cid, img, cond)}) + "\n")
                n += 1
                if n % 500 == 0:
                    fh.flush()
                    print(f"[pmc] eval {stage} {n}/{total} {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"[pmc] eval {stage} done {n} rows {(time.time() - t0) / 60:.1f} min", flush=True)


lim = 20 if args.smoke else 0
from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, args.adapter + "/adapter")
evaluate("a15", [("E1", IDS["E1"], args.arms.split(","))], lim, args.cond)
print("[a15] finished", flush=True)
print("RUN_DIR", run_dir, flush=True)
sys.stdout.flush()
os._exit(0)
