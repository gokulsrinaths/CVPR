#!/usr/bin/env python
"""A16: seed replication of the A11 PMC-VQA recipe for the 7B model (optional gradient checkpointing), derived from pmc_run3.py (A14). Original: A14 seed replication of the A11 PMC-VQA recipe (derived from pmc_run.py). Original: A11 PMC-VQA (no option-length cue) controlled study. zero-shot eval -> LoRA fine-tune on TR -> eval. Recipe identical to A9.
python pmc_run.py --model q3b|q7b|ivl2b [--smoke] | --selftest --model q3b   Never overwrites a run."""
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
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--gc", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SPL = ROOT + "/data/pmcvqa/split_a11"
MODEL = {"q3b": "qwen25vl_3b_instruct", "q7b": "qwen25vl_7b_instruct", "ivl2b": "internvl3_2b_hf"}[args.model]
MODEL = f"{ROOT}/models/{MODEL}"
SEED, ACC, LR, MAXSIDE = args.seed, 16, 1e-4, 448
LETTERS = "ABCD"
SYS = "You are a medical imaging expert."
INSTR = "Answer with only the letter (A, B, C or D) of the correct option."
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

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


def user_text(cid):
    r = R[cid]
    return r["question"] + "\n\nOptions:\n" + "\n".join(f"  {l}: {c}" for l, c in zip(LETTERS, r["choices"])) + "\n\n" + INSTR


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
torch.manual_seed(args.seed)
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

jobid = os.environ.get("SLURM_JOB_ID", "nojob")
tag = f"pmc16_{args.model}{'_smoke' if args.smoke else ''}_seed{args.seed}"
run_dir = Path(ROOT) / "experiments" / "pilot" / f"{tag}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"amendment": "A16", "seed": args.seed, "gradient_checkpointing": args.gc, "model": MODEL, "smoke": args.smoke, "script_sha256": sha(__file__), "amendment_sha256": sha(ROOT + "/docs/preregistration_v1_amendment_A16.md"),
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


def prompt_inputs(cid, img):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(cid)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=[text], images=[img], return_tensors="pt")


@torch.inference_mode()
def lp_letters(cid, img):
    inp = prompt_inputs(cid, img).to(dev)
    lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    letters = [float(lp[i]) for i in LID]
    return {"lp": letters, "mass": float(sum(math.exp(x) for x in letters))}


def evaluate(stage, plan, limit=0):
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
                else:
                    img = load_img(DON[g]["rand" if a == "mismatch_rand" else "nn"][cid])
                fh.write(json.dumps({"stage": stage, "group": g, "case_id": cid, "arm": a, "gold": R[cid]["gold"], "direct": lp_letters(cid, img)}) + "\n")
                n += 1
                if n % 500 == 0:
                    fh.flush()
                    print(f"[pmc] eval {stage} {n}/{total} {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"[pmc] eval {stage} done {n} rows {(time.time() - t0) / 60:.1f} min", flush=True)


lim = 20 if args.smoke else 0
ARMS4 = ["base", "black", "mismatch_rand", "mismatch_nn"]
PLAN = [("VAL700", IDS["VAL700"], ["base"]), ("E1", IDS["E1"], ARMS4), ("E2", IDS["E2"], ARMS4)]
evaluate("zeroshot", PLAN, lim)

from peft import LoraConfig, get_peft_model  # noqa: E402

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
if args.gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
model = get_peft_model(model, cfg)
ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[pmc] trainable parameters: {ntrain:,}", flush=True)
assert ntrain > 1_000_000
order = IDS["TR"][:]
random.Random(SEED).shuffle(order)
if args.smoke:
    order = order[:640]
total_steps = math.ceil(len(order) / ACC)
warm = max(1, int(0.03 * total_steps))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
model.train()
tl = open(run_dir / "train_log.jsonl", "a")
t0, acc_loss, k = time.time(), 0.0, 0
for i, cid in enumerate(order):
    inp = prompt_inputs(cid, load_img(cid))
    ids = inp["input_ids"]
    ids2 = torch.cat([ids, torch.tensor([[LID[LETTERS.index(R[cid]["gold"])], IM_END]])], dim=1).to(dev)
    labels = torch.full_like(ids2, -100)
    labels[0, -2:] = ids2[0, -2:]
    extra = {kk: v.to(dev) for kk, v in inp.items() if kk in ("pixel_values", "image_grid_thw")}
    out = model(input_ids=ids2, attention_mask=torch.ones_like(ids2), labels=labels, use_cache=False, **extra)
    loss = out.loss
    if not torch.isfinite(loss):
        print("[pmc] ABORT: non-finite loss at example", i, flush=True)
        sys.exit(5)
    (loss / ACC).backward()
    acc_loss += float(loss)
    k += 1
    if (i + 1) % ACC == 0 or i == len(order) - 1:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    if (i + 1) % 160 == 0 or i == len(order) - 1:
        el = time.time() - t0
        tl.write(json.dumps({"examples": i + 1, "mean_loss": acc_loss / k, "lr": sched.get_last_lr()[0], "minutes": round(el / 60, 2)}) + "\n")
        tl.flush()
        print(f"[pmc] train {i + 1}/{len(order)} loss {acc_loss / k:.4f} {el / 60:.1f} min ETA {el / (i + 1) * (len(order) - i - 1) / 60:.1f} min", flush=True)
        acc_loss, k = 0.0, 0
model.save_pretrained(str(run_dir / "adapter"))
evaluate("ft", PLAN, lim)
print("[pmc] finished", flush=True)
print("RUN_DIR", run_dir, flush=True)
sys.stdout.flush()
os._exit(0)
