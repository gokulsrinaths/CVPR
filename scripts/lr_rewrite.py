#!/usr/bin/env python
"""A6 step 1: length-equalising rewrite of the options of TEST711 + DEV700 with Qwen2.5-VL-7B-Instruct used as a text-only model.
The rewriter never sees the question, the image or any label. Two attempts (greedy, then sampled for failures). Never overwrites a run."""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_common as C

if len(sys.argv) > 1 and sys.argv[1] == "selftest":
    groups, T = C.load_cases()
    print("groups", {g: len(v) for g, v in groups.items()})
    assert len(groups["test711"]) == 711 and len(groups["dev700"]) == 700
    ex = {o.label: o.text for o in groups["test711"][0].options}
    print(C.build_prompt(ex)[:300])
    assert os.path.exists(C.ROOT + "/models/qwen25vl_7b_instruct/config.json")
    print("SELFTEST ALL PASSED")
    sys.exit(0)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = C.ROOT + "/models/qwen25vl_7b_instruct"
BATCH = 12
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
run_dir = os.path.join(C.ROOT, "experiments", "pilot", f"lr_rewrite_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}")
os.makedirs(run_dir)
groups, T = C.load_cases()
proc = AutoProcessor.from_pretrained(MODEL)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
json.dump({"amendment": "A6", "model": MODEL, "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "common_sha256": hashlib.sha256(open(C.__file__.replace(".pyc", ".py"), "rb").read()).hexdigest(),
           "amendment_sha256": hashlib.sha256(open(C.ROOT + "/docs/preregistration_v1_amendment_A6.md", "rb").read()).hexdigest(),
           "batch": BATCH, "sys": C.SYS, "attempt2": {"temperature": 0.7, "top_p": 1.0}}, open(os.path.join(run_dir, "config.json"), "w"), indent=2)


def gen(items, sample):
    """items: list of (key, orig). Returns {key: raw}. Batches of similar length."""
    out = {}
    items = sorted(items, key=lambda x: C.target(x[1])[0])
    for s in range(0, len(items), BATCH):
        b = items[s:s + BATCH]
        texts = []
        for _, orig in b:
            msgs = [{"role": "system", "content": C.SYS}, {"role": "user", "content": [{"type": "text", "text": C.build_prompt(orig)}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        Tmax = max(C.target(o)[0] for _, o in b)
        mx = min(900, 60 + int(5 * Tmax * 2.4))
        inp = proc(text=texts, padding=True, return_tensors="pt").to(model.device)
        kw = {"do_sample": False}
        if sample:
            torch.manual_seed(int(hashlib.sha256("|".join(k for k, _ in b).encode()).hexdigest()[:8], 16))
            kw = {"do_sample": True, "temperature": 0.7, "top_p": 1.0}
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=mx, **kw)
        dec = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for (k, _), r in zip(b, dec):
            out[k] = r
        print(f"[lr] {'sampled' if sample else 'greedy'} {min(s + BATCH, len(items))}/{len(items)}", flush=True)
    return out


res = {}
allitems = []
for g, cases in groups.items():
    for c in cases:
        orig = {o.label: o.text for o in c.options}
        allitems.append((f"{g}|{c.case_id}", orig))
origs = dict(allitems)
t0 = time.time()
raw1 = gen(allitems, False)
for k, orig in allitems:
    new = C.parse_options(raw1[k])
    ok, why = C.passes(orig, new)
    res[k] = {"orig": orig, "T": C.target(orig)[0], "attempt": 1, "raw": raw1[k], "new": new, "passed": ok, "reasons": why}
fails = [(k, origs[k]) for k in res if not res[k]["passed"]]
print(f"[lr] attempt 1: {len(res) - len(fails)}/{len(res)} pass; retrying {len(fails)}", flush=True)
if fails:
    raw2 = gen(fails, True)
    for k, orig in fails:
        new = C.parse_options(raw2[k])
        ok, why = C.passes(orig, new)
        res[k].update({"attempt1_raw": res[k]["raw"], "attempt1_reasons": res[k]["reasons"], "attempt": 2, "raw": raw2[k], "new": new, "passed": ok, "reasons": why})
json.dump(res, open(os.path.join(run_dir, "rewrites.json"), "w"))
print(f"[lr] finished {round(time.time() - t0)}s; pass {sum(v['passed'] for v in res.values())}/{len(res)}", flush=True)
print("RUN_DIR", run_dir, flush=True)
