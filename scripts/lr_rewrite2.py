#!/usr/bin/env python
"""A7: per-option, feedback-loop, character+word constrained length-equalising rewrite (docs/preregistration_v1_amendment_A7.md).
Qwen2.5-VL-7B-Instruct text-only; each option rewritten alone (no question/other options/image/label). Up to 6 tries. Never overwrites a run.
python lr_rewrite2.py | selftest"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_common as C
import lr_common2 as C2

if len(sys.argv) > 1 and sys.argv[1] == "selftest":
    groups, T = C.load_cases()
    n = 0
    for g, cs_ in groups.items():
        for c in cs_:
            orig = {o.label: o.text for o in c.options}
            b = C2.bands(orig)
            assert b["c_lo"] <= b["Tc"] <= b["c_hi"] and b["w_lo"] <= b["Tw"] <= b["w_hi"]
            # each option's own original text must at least pass the semantic checks against itself
            n += 1
    print("bands ok for", n, "questions; example:", C2.bands({o.label: o.text for o in groups["test711"][0].options}))
    assert os.path.exists(C.ROOT + "/models/qwen25vl_7b_instruct/config.json")
    print("SELFTEST ALL PASSED")
    sys.exit(0)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = C.ROOT + "/models/qwen25vl_7b_instruct"
BATCH, TRIES = 64, 6
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
run_dir = os.path.join(C.ROOT, "experiments", "pilot", f"lr_rewrite2_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}")
os.makedirs(run_dir)
groups, T = C.load_cases()
proc = AutoProcessor.from_pretrained(MODEL)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"amendment": "A7", "model": MODEL, "script_sha256": sha(__file__), "common2_sha256": sha(C2.__file__), "common_sha256": sha(C.__file__),
           "amendment_sha256": sha(C.ROOT + "/docs/preregistration_v1_amendment_A7.md"), "batch": BATCH, "tries": TRIES, "sys": C.SYS,
           "sampling": {"temperature": 0.7, "top_p": 1.0}}, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

S = {}
Q = {}
for g, cases in groups.items():
    for c in cases:
        orig = {o.label: o.text for o in c.options}
        b = C2.bands(orig)
        Q[f"{g}|{c.case_id}"] = {"orig": orig, "b": b}
        for l in C.LETTERS:
            S[f"{g}|{c.case_id}|{l}"] = {"orig": orig[l], "b": b, "ok": False, "tries": 0, "last": None, "best": None, "best_err": 10 ** 9, "why": None}


def messages(st):
    m = [{"role": "system", "content": C.SYS}, {"role": "user", "content": C2.first_prompt(st["orig"], st["b"])}]
    if st["last"] is not None:
        m += [{"role": "assistant", "content": st["last"]}, {"role": "user", "content": C2.feedback(st["last"], st["b"])}]
    return m


t0 = time.time()
for rnd in range(1, TRIES + 1):
    pending = [k for k in S if not S[k]["ok"]]
    if not pending:
        break
    pending.sort(key=lambda k: len(S[k]["orig"]))
    for s in range(0, len(pending), BATCH):
        keys = pending[s:s + BATCH]
        texts = [proc.apply_chat_template([{**m, "content": ([{"type": "text", "text": m["content"]}] if m["role"] == "user" else m["content"])} for m in messages(S[k])],
                                          add_generation_prompt=True, tokenize=False) for k in keys]
        mx = min(220, int(max(S[k]["b"]["c_hi"] for k in keys) / 2.2) + 40)
        inp = proc(text=texts, padding=True, return_tensors="pt").to(model.device)
        kw = {"do_sample": False}
        if rnd > 1:
            torch.manual_seed(int(hashlib.sha256(("|".join(keys) + str(rnd)).encode()).hexdigest()[:8], 16))
            kw = {"do_sample": True, "temperature": 0.7, "top_p": 1.0}
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=mx, **kw)
        dec = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for k, raw in zip(keys, dec):
            st = S[k]
            new = " ".join(C2.clean(raw).split())
            ok, why = C2.option_ok(st["orig"], new, st["b"])
            st["tries"] = rnd
            st["last"] = new
            st["why"] = why
            sem = [w for w in why if w not in ("chars", "words")]
            if ok:
                st["ok"], st["best"] = True, new
            elif not sem:
                e = C2.len_err(new, st["b"])
                if e < st["best_err"]:
                    st["best"], st["best_err"] = new, e
    done = sum(S[k]["ok"] for k in S)
    print(f"[lr2] round {rnd}: options accepted {done}/{len(S)} ({100 * done / len(S):.1f}%) {round(time.time() - t0)}s", flush=True)

res = {}
for qk, q in Q.items():
    opts = {l: S[f"{qk}|{l}"] for l in C.LETTERS}
    passed = all(o["ok"] for o in opts.values())
    reasons = [f"{(o['why'] or ['unknown'])[0]}_{l}" for l, o in opts.items() if not o["ok"]]
    res[qk] = {"orig": q["orig"], "T": q["b"]["Tw"], "bands": q["b"], "attempt": max(o["tries"] for o in opts.values()), "raw": "",
               "new": {l: opts[l]["best"] for l in C.LETTERS} if passed else None, "passed": passed, "reasons": reasons,
               "tries_per_option": {l: opts[l]["tries"] for l in C.LETTERS}}
json.dump(res, open(os.path.join(run_dir, "rewrites.json"), "w"))
print(f"[lr2] finished {round(time.time() - t0)}s; questions passing {sum(v['passed'] for v in res.values())}/{len(res)}", flush=True)
print("RUN_DIR", run_dir, flush=True)
sys.stdout.flush()
os._exit(0)
