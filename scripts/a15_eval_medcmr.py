#!/usr/bin/env python
"""A15: adapter-only evaluation of Med-CMR models with blurred-image arms; derived from ft_run4.py (A12). Original docstring: A12 length intervention and saturation-aware training. Qwen2.5-VL 3B or 7B. Never overwrites a run.
python ft_run4.py --model 3b|7b --mode l1|l2|tx|satw [--fold A|B] [--adapter RUN_DIR] [--weights FILE] [--smoke] | --selftest --model 3b --mode l2
 l1  : no training; load an A9 adapter and evaluate on option-length conditions orig/pres/rand (test-time intervention)
 l2  : train on the length-randomised training set, evaluate on randomised test/fresh
 tx  : text-only bias model: train on half of TR (fold) with a black image, score the other half (black image)
 satw: train as A9 but with per-question loss weights (from the cross-fitted text-only margins)"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["3b", "7b", "ivl2b"], required=True)
ap.add_argument("--mode", choices=["a15"], required=True)
ap.add_argument("--arms", default="base,blur8,blur24")
ap.add_argument("--fold", choices=["A", "B"])
ap.add_argument("--adapter")
ap.add_argument("--weights")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
MODEL = f"{ROOT}/models/" + {"3b": "qwen25vl_3b_instruct", "7b": "qwen25vl_7b_instruct", "ivl2b": "internvl3_2b_hf"}[args.model]
SEED, ACC, LR, MAXSIDE = 20261003, 16, 1e-4, 448
LAB = re.compile(r"^\s*[A-E]\)\s*")
LETTERS = "ABCDE"
SYS = "You are a medical imaging expert."
INSTR = "Answer with only the letter (A, B, C, D or E) of the correct option."
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_rand import pad_options  # noqa: E402

T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}
V = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_validation_participant_facing.json", encoding="utf-8"))["cases"]}
IDS = json.load(open(POOLD + "/derived/ids.json"))
H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}
NT = set(json.load(open(POOLD + "/split_v1/n_test_ids.json")))
fresh_run = glob.glob(ROOT + "/experiments/pilot/condrel_fresh_s0_22567079_*")[0]
FCFG = json.load(open(fresh_run + "/config.json"))
FRESH = FCFG["sample_ids"]
NN, RAND = FCFG["nn_donors"], FCFG["rand_donors"]
DEV700 = set(json.load(open(glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/groups.json")[0]))["dev700"])
J1G = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*/groups.json")[0]))
POOL700, VAL700 = set(J1G["pool700"]), J1G["val700"]
TEST = sorted(c for c in H if T[c]["question type"] == "mcq")
TR = [c for c in IDS if T[c]["question type"] == "mcq" and c not in H and c not in NT and c not in DEV700 and c not in POOL700 and c not in set(FRESH)]


def rec(cid):
    return T[cid] if cid in T else V[cid]


def img_path(cid):
    split = "train" if cid in T else "val"
    return f"{POOLD}/extracted/{split}/imgs/" + os.path.basename(rec(cid)["image_path"])


def load_img(cid):
    im = Image.open(img_path(cid)).convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s < 1 else im


def user_text(cid, cond="orig"):
    r = rec(cid)
    opts = pad_options(cid, [LAB.sub("", r[l]).strip() for l in LETTERS], cond)
    return r["question"].strip() + "\n\nOptions:\n" + "\n".join(f"  {l}: {o}" for l, o in zip(LETTERS, opts)) + "\n\n" + INSTR


def fold_of(cid):
    return "A" if int(hashlib.sha256(("A12fold" + cid).encode()).hexdigest()[:8], 16) % 2 == 0 else "B"


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    print("     sizes: TR", len(TR), "| TEST", len(TEST), "| FRESH", len(FRESH), "| VAL700", len(VAL700), flush=True)
    check("TR is disjoint from test, N(T), DEV700, POOL700, FRESH3000", not (set(TR) & (H | NT | DEV700 | POOL700 | set(FRESH))))
    check("TEST has 711 MCQs, FRESH has 3000, VAL700 has 700", len(TEST) == 711 and len(FRESH) == 3000 and len(VAL700) == 700)
    check("all val ids are in the validation file and not in train", all(c in V and c not in T for c in VAL700))
    check("TR is large and MCQ-only", len(TR) > 5000 and all(T[c]["question type"] == "mcq" for c in TR))
    check("donor maps cover FRESH", all(c in NN and c in RAND for c in FRESH))
    check("images exist (sample)", all(os.path.exists(img_path(c)) for c in (TR[:200] + TEST[:100] + FRESH[:100] + VAL700[:100] + [NN[c] for c in FRESH[:100]])))
    im = load_img(TR[0])
    check("image downscaled to <= 448 on the long side", max(im.size) <= MAXSIDE)
    check("user text format", user_text(TR[0]).endswith(INSTR) and "Options:\n  A: " in user_text(TR[0]))
    check("model dir present", os.path.exists(MODEL + "/config.json"))
    ur = user_text(TR[0], "rand")
    check("rand text differs and ends with the instruction", ur != user_text(TR[0]) and ur.endswith(INSTR))
    fa = [c for c in TR if fold_of(c) == "A"]
    check("folds roughly balanced", 0.45 < len(fa) / len(TR) < 0.55)
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


if args.selftest:
    selftest()

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

jobid = os.environ.get("SLURM_JOB_ID", "nojob")
tag = f"a15med_{args.model}{'_smoke' if args.smoke else ''}"
run_dir = Path(ROOT) / "experiments" / "pilot" / f"{tag}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"amendment": "A15", "arms": args.arms, "mode": args.mode, "fold": args.fold, "adapter_in": args.adapter, "weights": args.weights, "model": MODEL, "smoke": args.smoke, "script_sha256": sha(__file__), "amendment_sha256": sha(ROOT + "/docs/preregistration_v1_amendment_A15.md"),
           "recipe": {"lora_r": 16, "alpha": 32, "dropout": 0.05, "lr": LR, "accum": ACC, "epochs": 1, "seed": SEED, "max_side": MAXSIDE}, "sizes": {"TR": len(TR), "TEST": len(TEST), "FRESH": len(FRESH), "VAL700": len(VAL700)},
           "fresh_run_config": fresh_run}, open(run_dir / "config.json", "w"), indent=2)
proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
dev = model.device
LID = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
assert all(len(x) == 1 for x in LID), LID
LID = [x[0] for x in LID]
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
json.dump({"host": os.uname().nodename, "slurm_job_id": jobid, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "letter_ids": LID, "im_end": IM_END,
           "attn": getattr(model.config, "_attn_implementation", None)}, open(run_dir / "env.json", "w"), indent=2)


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
    path = run_dir / f"rows_{stage}.jsonl"
    fh = open(path, "a")
    n = 0
    t0 = time.time()
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
                    img = load_img(RAND[cid] if a == "mismatch_rand" else NN[cid])
                fh.write(json.dumps({"stage": stage, "cond": cond, "group": g, "case_id": cid, "arm": a, "gold": T[cid]["answer"].strip() if cid in T else None, "direct": lp_letters(cid, img, cond)}) + "\n")
                n += 1
                if n % 500 == 0:
                    fh.flush()
                    print(f"[ft] eval {stage} {n}/{total} {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"[ft] eval {stage} done {n} rows {(time.time() - t0) / 60:.1f} min", flush=True)


lim = 20 if args.smoke else 0
from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, args.adapter + "/adapter")
evaluate("a15", [("fresh", FRESH, args.arms.split(","))], lim)
print("[a15] finished", flush=True)
print("RUN_DIR", run_dir, flush=True)
sys.stdout.flush()
os._exit(0)
