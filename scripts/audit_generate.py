#!/usr/bin/env python
"""H1' (amendment A3): base Qwen3.5-9B, NO adapter, greedy generated answer with enable_thinking=False on the 711 test MCQs,
arms true image / black image. Prompt = the frozen pipeline's, byte for byte. Resumable; never overwrites a run."""
import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=None)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
os.environ["MEDREASON_ADAPTER_PATH"] = "/nonexistent/no_adapter_gateB"  # before importing the system module
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvqa_diag as md  # noqa: E402

cs = md.cs
POOLD = ROOT + "/data/train_pool"
LETTER = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}


def parse(raw):
    m = LETTER.search(raw)
    return m.group(1) if m else None


def main():
    cases = md.load_mcq_cases()
    if args.limit:
        cases = cases[: args.limit]
    assert len(cases) == (args.limit or 711)
    gold = {c.case_id: T[c.case_id]["answer"].strip() for c in cases}
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"gateB_h1gen_base_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        json.dump({"job": "h1gen", "amendment": "A3", "script_sha256": md.sha256_file(__file__),
                   "mvqa_diag_sha256": md.sha256_file(ROOT + "/scripts/mvqa_diag.py"),
                   "amendment_sha256": md.sha256_file(ROOT + "/docs/preregistration_v1_amendment_A3.md"),
                   "decode": {"do_sample": False, "max_new_tokens": 16, "repetition_penalty": 1.0, "enable_thinking": False},
                   "arms": ["base", "black"], "n": len(cases)}, open(run_dir / "config.json", "w"), indent=2)
    system = cs.CustomMedReasonSystem(md.make_config())
    system.setup()
    torch, device = system.torch, system.model.device
    adapter_loaded = hasattr(system.model, "peft_config")
    info = md.env_info(system)
    info["adapter_loaded"] = adapter_loaded
    json.dump(info, open(run_dir / "env.json", "w"), indent=2)
    print("[h1gen] adapter_loaded=", adapter_loaded, flush=True)
    if adapter_loaded:
        sys.exit(5)
    tok = system.processor.tokenizer
    rows_p = run_dir / "rows.jsonl"
    done = {(r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
    fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")
    t0, n_err = time.time(), 0
    for ci, case in enumerate(cases):
        need = [a for a in ("base", "black") if (case.case_id, a) not in done]
        if not need:
            continue
        try:
            p = case.primary_image_path
            rgb = np.array(Image.open(p).convert("RGB"))
            modality = system._detect(rgb)
            image = Image.fromarray(cs._preprocess(rgb, modality))
            for arm in need:
                img = image if arm == "base" else Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                messages = [{"role": "system", "content": cs.SYSTEM_PROMPT + cs.MAP_PROMPTS.get(modality, "")},
                            {"role": "user", "content": [{"type": "image", "image": img},
                                                         {"type": "text", "text": system._user_text(case)}]}]
                text = system.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = system.processor(text=[text], images=[[img]], return_tensors="pt").to(device)
                n_prompt = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    out = system.model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
                raw = system.processor.decode(out[0, n_prompt:], skip_special_tokens=True)
                md.append_jsonl(fh, {"case_id": case.case_id, "arm": arm, "gold": gold[case.case_id], "raw": raw,
                                     "parsed": parse(raw), "modality_route": modality})
        except Exception:  # noqa: BLE001
            n_err += 1
            md.append_jsonl(eh, {"case_id": case.case_id, "traceback": traceback.format_exc()})
            print("[h1gen] ERROR", case.case_id, traceback.format_exc(), flush=True)
            if ci == 0 or n_err > 10:
                sys.exit(4)
        if (ci + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[h1gen] {ci + 1}/{len(cases)} {el / 60:.1f} min ETA {el / (ci + 1) * (len(cases) - ci - 1) / 60:.1f} min", flush=True)
    print("[h1gen] finished", flush=True)


if __name__ == "__main__":
    main()
