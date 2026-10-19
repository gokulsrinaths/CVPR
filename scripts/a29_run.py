#!/usr/bin/env python
"""A29: lock-out prediction on OmniMedVQA. Plants a nonvisual shortcut (option length via neutral filler, or answer position) at a controlled strength in training AND matched evaluation
options, fine-tunes with the A11 recipe, and evaluates with true/black/wrong images and a ladder of degraded images. Never overwrites a run.
python dose_run.py --family q3b|q7b|ivl2b --shortcut length|position --dose P --seed S [--arms a,b,...] [--gc] [--smoke] | --selftest ..."""
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
ap.add_argument("--family", choices=["q3b", "q7b", "ivl2b"], required=True)
ap.add_argument("--shortcut", choices=["length", "position"], required=True)
ap.add_argument("--dose", type=float, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arms", default="base,black,blur32,blur16,blur8,blur4,blur2")
ap.add_argument("--gc", action="store_true")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SPL = ROOT + "/data/omnimed/split_a16"
MODEL = ROOT + "/models/" + {"q3b": "qwen25vl_3b_instruct", "q7b": "qwen25vl_7b_instruct", "ivl2b": "internvl3_2b_hf"}[args.family]
SEED, ACC, LR, MAXSIDE = args.seed, 16, 1e-4, 448
LETTERS = "ABCD"
SYS = "You are a medical imaging expert."
INSTR = "Answer with only the letter (A, B, C or D) of the correct option."
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dose_plant import assign_u, length_options, position_options  # noqa: E402

REC = json.load(open(SPL + "/records.json"))
IDS = json.load(open(SPL + "/ids.json"))
DON = json.load(open(SPL + "/donors.json"))
R = {}
for g in REC.values():
    R.update(g)
U = {}
for g in ("TR", "VAL700", "E1"):
    U.update(assign_u(IDS[g]))
PR = {}
for g in ("TR", "VAL700", "E1"):
    for cid in IDS[g]:
        r = R[cid]
        gi = LETTERS.index(r["gold"])
        if args.shortcut == "length":
            PR[cid] = {"choices": length_options(cid, r["choices"], gi, U[cid], args.dose), "gold": r["gold"]}
        else:
            ch, gl = position_options(cid, r["choices"], gi, U[cid], args.dose)
            PR[cid] = {"choices": ch, "gold": gl}


def rec_of(cid, cond):
    return PR[cid] if cond == "planted" else R[cid]


def load_img(cid):
    im = Image.open(R[cid]["image"]).convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s < 1 else im


def user_text(cid, cond):
    r = rec_of(cid, cond)
    return R[cid]["question"] + "\n\nOptions:\n" + "\n".join(f"  {l}: {c}" for l, c in zip(LETTERS, r["choices"])) + "\n\n" + INSTR


def degrade(real, a):
    if a.startswith("blur"):
        return real.filter(ImageFilter.GaussianBlur(int(a[4:])))
    if a.startswith("down"):
        w, h = real.size
        sc = int(a[4:]) / max(w, h)
        return real.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    raise ValueError(a)


def realized_rate(group):
    ids = IDS[group]
    if args.shortcut == "length":
        n = 0
        for c in ids:
            L = [len(x) for x in PR[c]["choices"]]
            g = LETTERS.index(PR[c]["gold"])
            n += (L[g] == min(L) and L.count(min(L)) == 1)
        return n / len(ids)
    return sum(PR[c]["gold"] == "A" for c in ids) / len(ids)


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    check("sizes", [len(IDS[g]) for g in ("E1", "VAL700", "TR")] == [2000, 700, 9277])
    for g in ("TR", "E1"):
        rr = realized_rate(g)
        print(f"     realized {args.shortcut} cue rate on {g}: {rr:.4f} (target {args.dose})")
        check(f"realized cue rate on {g} equals the dose to within 0.002", abs(rr - args.dose) <= 0.002)
    c = IDS["TR"][0]
    check("question text unchanged by planting", user_text(c, "planted").split("\n\nOptions:")[0] == user_text(c, "orig").split("\n\nOptions:")[0])
    check("option multiset preserved (position) / text is prefix (length)", all((sorted(PR[c]["choices"]) == sorted(R[c]["choices"])) if args.shortcut == "position"
                                                                               else all(PR[c]["choices"][k].startswith(R[c]["choices"][k]) for k in range(4)) for c in IDS["TR"][:300]))
    check("gold text preserved", all((PR[c]["choices"][LETTERS.index(PR[c]["gold"])].startswith(R[c]["choices"][LETTERS.index(R[c]["gold"])])) for c in IDS["TR"][:300]))
    im = load_img(IDS["E1"][0])
    for a in ("blur32", "blur2", "down8", "down128"):
        check(f"degradation {a} keeps the image size", degrade(im, a).size == im.size)
    check("model dir", os.path.exists(MODEL + "/config.json"))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


if args.selftest:
    selftest()

import torch  # noqa: E402
torch.manual_seed(args.seed)
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

jobid = os.environ.get("SLURM_JOB_ID", "nojob")
tag = f"a29_{args.family}_{args.shortcut}{args.dose:.2f}_seed{args.seed}{'_smoke' if args.smoke else ''}"
run_dir = Path(ROOT) / "experiments" / "pilot" / f"{tag}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"amendment": "A29", "shortcut": args.shortcut, "dose": args.dose, "seed": args.seed, "family": args.family, "model": MODEL, "arms": args.arms, "gradient_checkpointing": args.gc, "smoke": args.smoke,
           "script_sha256": sha(__file__), "amendment_sha256": sha(ROOT + "/docs/preregistration_v1_amendment_A29.md"), "split_sha256": json.load(open(SPL + "/SHA256.json")),
           "realized_rate_TR": realized_rate("TR"), "realized_rate_E1": realized_rate("E1"),
           "recipe": {"lora_r": 16, "alpha": 32, "dropout": 0.05, "lr": LR, "accum": ACC, "epochs": 1, "seed": SEED, "max_side": MAXSIDE}}, open(run_dir / "config.json", "w"), indent=2)
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


def prompt_inputs(cid, img, cond):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(cid, cond)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=[text], images=[img], return_tensors="pt")


@torch.inference_mode()
def lp_letters(cid, img, cond):
    inp = prompt_inputs(cid, img, cond).to(dev)
    lp = torch.log_softmax(model(**inp).logits[0, -1].float(), -1)
    letters = [float(lp[i]) for i in LID]
    return {"lp": letters, "mass": float(sum(math.exp(x) for x in letters))}


def evaluate(stage, plan, cond, limit=0):
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
                elif a.startswith(("blur", "down")):
                    img = degrade(real, a)
                else:
                    img = load_img(DON[g]["rand" if a == "mismatch_rand" else "nn"][cid])
                fh.write(json.dumps({"stage": stage, "cond": cond, "group": g, "case_id": cid, "arm": a, "gold": rec_of(cid, cond)["gold"], "direct": lp_letters(cid, img, cond)}) + "\n")
                n += 1
                if n % 500 == 0:
                    fh.flush()
                    print(f"[dose] eval {stage} {n}/{total} {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"[dose] eval {stage} done {n} rows {(time.time() - t0) / 60:.1f} min", flush=True)


lim = 20 if args.smoke else 0
from peft import LoraConfig, get_peft_model  # noqa: E402

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
if args.gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
model = get_peft_model(model, cfg)
ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[dose] trainable parameters: {ntrain:,}", flush=True)
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
    inp = prompt_inputs(cid, load_img(cid), "planted")
    ids = inp["input_ids"]
    ids2 = torch.cat([ids, torch.tensor([[LID[LETTERS.index(PR[cid]["gold"])], IM_END]])], dim=1).to(dev)
    labels = torch.full_like(ids2, -100)
    labels[0, -2:] = ids2[0, -2:]
    extra = {kk: v.to(dev) for kk, v in inp.items() if kk in ("pixel_values", "image_grid_thw")}
    loss = model(input_ids=ids2, attention_mask=torch.ones_like(ids2), labels=labels, use_cache=False, **extra).loss
    if not torch.isfinite(loss):
        print("[dose] ABORT: non-finite loss at example", i, flush=True)
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
        print(f"[dose] train {i + 1}/{len(order)} loss {acc_loss / k:.4f} {el / 60:.1f} min ETA {el / (i + 1) * (len(order) - i - 1) / 60:.1f} min", flush=True)
        acc_loss, k = 0.0, 0
model.save_pretrained(str(run_dir / "adapter"))
arms = args.arms.split(",")
evaluate("ft_planted", [("VAL700", IDS["VAL700"], ["base"]), ("E1", IDS["E1"], arms)], "planted", lim)
evaluate("ft_orig", [("E1", IDS["E1"], ["base", "black"])], "orig", lim)
print("[dose] finished", flush=True)
print("RUN_DIR", run_dir, flush=True)
sys.stdout.flush()
os._exit(0)
