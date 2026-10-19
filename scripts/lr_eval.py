#!/usr/bin/env python
"""A6 step 3: evaluate the frozen adapter pipeline on the length-equalised options (arms eq, eq_black), teacher-forced direct+cue readouts.
Refuses to run unless the hand check (G2) exists with >= 45/50 preserved. python lr_eval.py --rewrite-run DIR [--run-dir DIR]"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--rewrite-run", required=True)
ap.add_argument("--run-dir", default=None)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import lr_common as C  # noqa: E402
import mvqa_diag as md  # noqa: E402
from medreason_docker.schema import MedReasonOption  # noqa: E402

cs = md.cs
ROOT = C.ROOT
hc = json.load(open(ROOT + "/analysis/lr_handcheck.json"))
assert hc["n"] == 50 and hc["preserved"] >= 45, f"G2 not met: {hc['preserved']}/50"
eq = json.load(open(os.path.join(args.rewrite_run, "eq_options.json")))
groups, T = C.load_cases()
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
resume = args.run_dir is not None
run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"lr_eval_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=resume)
if not resume:
    json.dump({"amendment": "A6", "rewrite_run": args.rewrite_run, "handcheck_preserved": hc["preserved"], "script_sha256": md.sha256_file(__file__),
               "eq_options_sha256": md.sha256_file(os.path.join(args.rewrite_run, "eq_options.json")),
               "handcheck_sha256": md.sha256_file(ROOT + "/analysis/lr_handcheck.json"),
               "n_passing": {g: len(v) for g, v in eq.items()}}, open(run_dir / "config.json", "w"), indent=2)
system = cs.CustomMedReasonSystem(md.make_config())
system.setup()
torch, device = system.torch, system.model.device
assert hasattr(system.model, "peft_config"), "adapter must be ON"
json.dump(md.env_info(system), open(run_dir / "env.json", "w"), indent=2)
dids, cids = md.letter_token_ids(system.processor.tokenizer)
rows_p = run_dir / "rows.jsonl"
done = {(r["group"], r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")
total = sum(len(v) for v in eq.values())
t0, seen, n_err = time.time(), 0, 0
for g, cases in groups.items():
    for case in cases:
        if case.case_id not in eq[g]:
            continue
        seen += 1
        need = [a for a in ("eq", "eq_black") if (g, case.case_id, a) not in done]
        if need:
            try:
                new = eq[g][case.case_id]
                c2 = dataclasses.replace(case, options=tuple(MedReasonOption(label=l, text=new[l]) for l in C.LETTERS))
                rgb = np.array(Image.open(case.primary_image_path).convert("RGB"))
                modality = system._detect(rgb)
                image = Image.fromarray(cs._preprocess(rgb, modality))
                for arm in need:
                    img = image if arm == "eq" else Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                    kw, n_direct, _ = md.prepare(system, c2, img, modality, device, torch)
                    res = md.score_forward(system, kw, n_direct, dids, cids, torch)
                    md.append_jsonl(fh, {"group": g, "case_id": case.case_id, "arm": arm, "gold": T[case.case_id]["answer"].strip(), **res})
                    del kw
            except Exception:  # noqa: BLE001
                n_err += 1
                md.append_jsonl(eh, {"case_id": case.case_id, "traceback": traceback.format_exc()})
                print("[lreval] ERROR", case.case_id, traceback.format_exc(), flush=True)
                if seen == 1 or n_err > 10:
                    sys.exit(4)
        if seen % 100 == 0:
            el = time.time() - t0
            print(f"[lreval] {seen}/{total} {el / 60:.1f} min ETA {el / seen * (total - seen) / 60:.1f} min", flush=True)
print("[lreval] finished", flush=True)
