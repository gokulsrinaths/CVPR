#!/usr/bin/env python
"""A4: Qwen2.5-VL-7B-Instruct image-dependence audit (see docs/preregistration_v1_amendment_A4.md). Resumable; never overwrites a run.
  python audit_s2.py [--run-dir DIR] [--limit N] | --selftest  (CPU, no model)"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
MODEL = ROOT + "/models/qwen25vl_7b_instruct"
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvqa_diag as md  # noqa: E402  (helpers only; the Qwen3.5 system is never loaded)

SEED, N = 20261002, 700
LAB = re.compile(r"^\s*[A-E]\)\s*")
LETTER = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
SYS = "You are a medical imaging expert."
INSTR = "Answer with only the letter (A, B, C, D or E) of the correct option."
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}


def parse(raw):
    m = LETTER.search(raw)
    return m.group(1) if m else None


def user_text(question, options):
    return question.strip() + "\n\nOptions:\n" + "\n".join(f"  {l}: {t}" for l, t in options) + "\n\n" + INSTR


def groups():
    test = md.load_mcq_cases()
    G = {"test711": [(c.case_id, c.question, [(o.label, o.text) for o in c.options], str(c.primary_image_path)) for c in test]}
    dev = json.load(open(POOLD + "/split_v1/dev_ids.json"))
    dm = sorted(i for i in dev if T[i]["question type"] == "mcq")
    ds = sorted(random.Random(SEED).sample(dm, N))
    G["dev700"] = [(i, T[i]["question"], [(l, LAB.sub("", T[i][l]).strip()) for l in "ABCDE"],
                    POOLD + "/extracted/train/imgs/" + os.path.basename(T[i]["image_path"])) for i in ds]
    return G


def downscale(im, side=896):
    w, h = im.size
    s = side / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    G = groups()
    check("test711 has 711, dev700 has 700", len(G["test711"]) == 711 and len(G["dev700"]) == 700)
    h2 = glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/groups.json")
    check("dev700 ids equal H2's sample", bool(h2) and [x[0] for x in G["dev700"]] == json.load(open(h2[0]))["dev700"])
    check("all images exist", all(os.path.exists(x[3]) for g in G.values() for x in g))
    check("5 options A-E", all([l for l, _ in x[2]] == list("ABCDE") for g in G.values() for x in g))
    check("model dir has weights + config", os.path.isdir(MODEL) and bool(glob.glob(MODEL + "/*.safetensors")) and os.path.exists(MODEL + "/config.json"))
    check("parse: 'C' -> C", parse("C") == "C")
    check("parse: 'The answer is C.' -> C", parse("The answer is C.") == "C")
    check("parse: '(B)' -> B", parse("(B) text") == "B")
    check("parse: prose with no standalone letter -> None", parse("Based on the image, the lesion") is None)
    ut = user_text("Q?", [("A", "x"), ("B", "y")])
    check("user text format", ut.startswith("Q?\n\nOptions:\n  A: x\n  B: y\n\nAnswer with only the letter"))
    im = downscale(Image.new("RGB", (3000, 1500)))
    check("downscale longest side 896", max(im.size) == 896 and im.size == (896, 448))
    check("downscale never upscales", downscale(Image.new("RGB", (300, 200))).size == (300, 200))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


def run():
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    G = groups()
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"gateC_s2_qwen25vl_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        json.dump({"job": "s2", "amendment": "A4", "model": MODEL, "script_sha256": md.sha256_file(__file__),
                   "amendment_sha256": md.sha256_file(ROOT + "/docs/preregistration_v1_amendment_A4.md"),
                   "decode": {"do_sample": False, "max_new_tokens": 8}, "max_side_px": 896, "system": SYS, "instruction": INSTR,
                   "groups": {g: [x[0] for x in v] for g, v in G.items()}}, open(run_dir / "config.json", "w"), indent=2)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    json.dump({"host": os.uname().nodename, "slurm_job_id": jobid, "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "attn": getattr(model.config, "_attn_implementation", None),
               "model_class": type(model).__name__}, open(run_dir / "env.json", "w"), indent=2)
    rows_p = run_dir / "rows.jsonl"
    done = {(r["group"], r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
    fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")
    t0, n_err, seen = time.time(), 0, 0
    total = sum(len(v[: args.limit] if args.limit else v) for v in G.values())
    for g, cases in G.items():
        for cid, q, opts, ip in (cases[: args.limit] if args.limit else cases):
            seen += 1
            need = [a for a in ("base", "black") if (g, cid, a) not in done]
            if not need:
                continue
            try:
                real = downscale(Image.open(ip).convert("RGB"))
                for arm in need:
                    img = real if arm == "base" else Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                    messages = [{"role": "system", "content": SYS},
                                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(q, opts)}]}]
                    text = proc.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
                    n_prompt = inputs["input_ids"].shape[1]
                    with torch.inference_mode():
                        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    raw = proc.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0]
                    md.append_jsonl(fh, {"group": g, "case_id": cid, "arm": arm, "gold": T[cid]["answer"].strip(), "raw": raw,
                                         "parsed": parse(raw), "n_prompt_tokens": int(n_prompt)})
            except Exception:  # noqa: BLE001
                n_err += 1
                md.append_jsonl(eh, {"group": g, "case_id": cid, "traceback": traceback.format_exc()})
                print("[s2] ERROR", cid, traceback.format_exc(), flush=True)
                if seen == 1 or n_err > 10:
                    sys.exit(4)
            if seen % 50 == 0:
                el = time.time() - t0
                print(f"[s2] {seen}/{total} {el / 60:.1f} min ETA {el / seen * (total - seen) / 60:.1f} min", flush=True)
    print("[s2] finished", flush=True)


if __name__ == "__main__":
    selftest() if args.selftest else run()
