#!/usr/bin/env python
"""Gate B teacher-forced forward-pass audit (docs/gate_B_prespec.md). Reuses scripts/mvqa_diag.py helpers unmodified.

  --job validity : adapter ON.  Groups anchor20 / pool700 / val700, arm 'base'.  (development-pool validity check)
  --job h1       : adapter OFF. Group test711, arms 'base' and 'black', readouts direct + cue + empty-think.
  --selftest     : CPU only; builds the groups and checks them (no model).
Never overwrites a run: output goes to experiments/pilot/<tag>_<jobid>_<ts>/ (pass --run-dir to RESUME the same run).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--job", choices=["validity", "h1"], required=True)
ap.add_argument("--run-dir", default=None)
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="debug: first N cases per group")
args = ap.parse_args()

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
if args.job == "h1":  # must be set BEFORE importing the frozen system module (it reads the path at import)
    os.environ["MEDREASON_ADAPTER_PATH"] = "/nonexistent/no_adapter_gateB"
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvqa_diag as md  # noqa: E402
from medreason_docker.schema import MedReasonCase, MedReasonOption  # noqa: E402

cs = md.cs
POOLD = ROOT + "/data/train_pool"
LAB = re.compile(r"^\s*[A-E]\)\s*")
SEED = 20261002
N = 700
ARMS = {"validity": ["base"], "h1": ["base", "black"]}


def sha(p):
    return md.sha256_file(p)


def mk(rec, split):
    opts = tuple(MedReasonOption(label=l, text=LAB.sub("", rec[l]).strip()) for l in md.LETTERS)
    p = Path(POOLD) / "extracted" / split / "imgs" / os.path.basename(rec["image_path"])
    return MedReasonCase(case_id=rec["case_id"], task_type="mcq", question=rec["question"].strip(), image_paths=(p,),
                         options=opts, metadata={})


def build_groups(job):
    T = json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
    V = json.load(open(POOLD + "/official_json/medreason_validation_participant_facing.json", encoding="utf-8"))["cases"]
    H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}
    byT = {c["case_id"]: c for c in T}
    byV = {c["case_id"]: c for c in V}
    test = md.load_mcq_cases()
    gold = {c.case_id: byT[c.case_id]["answer"].strip() for c in test}
    G = {}
    if job == "validity":
        pool = sorted(c["case_id"] for c in T if c["question type"] == "mcq" and c["case_id"] not in H)
        val = sorted(c["case_id"] for c in V if c["question type"] == "mcq")
        rng = random.Random(SEED)
        ps, vs = sorted(rng.sample(pool, N)), sorted(rng.sample(val, N))
        G["anchor20"] = test[:20]
        G["pool700"] = [mk(byT[i], "train") for i in ps]
        G["val700"] = [mk(byV[i], "val") for i in vs]
        for i in ps:
            gold[i] = byT[i]["answer"].strip()
    else:
        G["test711"] = test
    return G, gold, H


def selftest():
    fails = []

    def check(name, ok):
        print(("ok   " if ok else "FAIL ") + name, flush=True)
        if not ok:
            fails.append(name)

    job = args.job
    G, gold, H = build_groups(job)
    G2, _, _ = build_groups(job)
    check(job + ": sampling is deterministic", {k: [c.case_id for c in v] for k, v in G.items()} == {k: [c.case_id for c in v] for k, v in G2.items()})
    for g, cases in G.items():
        ids = [c.case_id for c in cases]
        check(f"{job}/{g}: n={len(cases)} unique ids", len(set(ids)) == len(ids))
        check(f"{job}/{g}: images exist", all(c.primary_image_path.exists() for c in cases))
        check(f"{job}/{g}: 5 labelled non-empty options A-E", all([o.label for o in c.options] == list(md.LETTERS) and all(o.text for o in c.options) for c in cases))
    if job == "validity":
        check("sizes 20/700/700", [len(G[k]) for k in ("anchor20", "pool700", "val700")] == [20, 700, 700])
        pi, vi = {c.case_id for c in G["pool700"]}, {c.case_id for c in G["val700"]}
        check("pool sample disjoint from the 886 test ids", not (pi & H))
        check("pool and val samples disjoint", not (pi & vi))
        check("pool ids and val ids are MCQ_*", all(i.startswith("MCQ_") for i in pi | vi))
        check("gold known for every pool case", all(i in gold for i in pi))
        check("adapter directory present (adapter ON)", os.path.isdir(cs.ADAPTER_DIR))
    else:
        check("711 test MCQs", len(G["test711"]) == 711)
        check("adapter path absent (adapter OFF)", not os.path.isdir(cs.ADAPTER_DIR))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


def prepare_empty(system, case, image, modality, device, torch):
    text = md.build_text(system, case, image, modality)
    inputs = system.processor(text=[text + "\n</think>\n\n"], images=[[image]], return_tensors="pt")
    ids = inputs["input_ids"]
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k]
    return {k: v.to(device) for k, v in kw.items()}, int(ids.shape[1])


def score_empty(system, kw, n, dids, torch):
    with torch.inference_mode():
        out = system.model(**kw)
    lp = torch.log_softmax(out.logits[0][n - 1].float(), dim=-1)
    letters = [float(lp[i]) for i in dids]
    del out
    return {"lp": letters, "mass": float(sum(np.exp(letters)))}


def run():
    G, gold, H = build_groups(args.job)
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    tag = "gateB_validity_s0" if args.job == "validity" else "gateB_h1_base_s0"
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"{tag}_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        json.dump({"job": args.job, "args": vars(args), "seed": SEED, "n_per_group": N, "arms": ARMS[args.job],
                   "script_sha256": sha(__file__), "mvqa_diag_sha256": sha(ROOT + "/scripts/mvqa_diag.py"),
                   "prespec_sha256": sha(ROOT + "/docs/gate_B_prespec.md"),
                   "official_json_sha256": {os.path.basename(p): sha(p) for p in glob.glob(POOLD + "/official_json/*.json")},
                   "frozen_manifest_sha256": sha(f"{md.FROZEN}/MANIFEST.json"), "stub": md.STUB},
                  open(run_dir / "config.json", "w"), indent=2)
        json.dump({g: [c.case_id for c in cs_] for g, cs_ in G.items()}, open(run_dir / "groups.json", "w"))
    print(f"[gateB] job={args.job} run_dir={run_dir} resume={resume}", flush=True)

    system = cs.CustomMedReasonSystem(md.make_config())
    system.setup()
    torch, device = system.torch, system.model.device
    adapter_loaded = hasattr(system.model, "peft_config")
    info = md.env_info(system)
    info["adapter_loaded"] = adapter_loaded
    json.dump(info, open(run_dir / "env.json", "w"), indent=2)
    print("[gateB] env:", json.dumps(info), flush=True)
    if adapter_loaded != (args.job == "validity"):
        print(f"[gateB] ABORT: adapter_loaded={adapter_loaded} contradicts job {args.job}", flush=True)
        sys.exit(5)
    dids, cids = md.letter_token_ids(system.processor.tokenizer)

    rows_p = run_dir / "forward_rows.jsonl"
    done = {(r["group"], r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
    fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")

    def load_rgb(case):
        p = case.primary_image_path
        return np.array(Image.open(p).convert("RGB")) if p and Path(p).exists() else np.zeros((224, 224, 3), np.uint8)

    ga_rows = {}
    if args.job == "validity":
        for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
            for r in md.read_jsonl(pth):
                if r["arm"] == "base":
                    ga_rows[r["case_id"]] = r["direct"]["lp"]
        assert len(ga_rows) == 711, len(ga_rows)

    t0, n_err, total = time.time(), 0, sum(len(v) for v in G.values())
    seen = 0
    for g, cases in G.items():
        if args.limit:
            cases = cases[: args.limit]
        for case in cases:
            seen += 1
            need = [a for a in ARMS[args.job] if (g, case.case_id, a) not in done]
            if need:
                try:
                    rgb = load_rgb(case)
                    modality = system._detect(rgb)
                    image = Image.fromarray(cs._preprocess(rgb, modality))
                    for arm in need:
                        ts0 = time.time()
                        img = image if arm == "base" else Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                        kw, n_direct, _ = md.prepare(system, case, img, modality, device, torch)
                        res = md.score_forward(system, kw, n_direct, dids, cids, torch)
                        row = {"group": g, "case_id": case.case_id, "arm": arm, "gold": gold.get(case.case_id),
                               "modality_route": modality, "n_direct_tokens": n_direct, **res}
                        del kw
                        if args.job == "h1":
                            kw2, n2 = prepare_empty(system, case, img, modality, device, torch)
                            row["empty"] = score_empty(system, kw2, n2, dids, torch)
                            del kw2
                        row["seconds"] = round(time.time() - ts0, 3)
                        md.append_jsonl(fh, row)
                except Exception:  # noqa: BLE001
                    n_err += 1
                    md.append_jsonl(eh, {"case_id": case.case_id, "group": g, "traceback": traceback.format_exc()})
                    print(f"[gateB] ERROR {g}/{case.case_id}\n{traceback.format_exc()}", flush=True)
                    if n_err > max(5, 0.02 * total):
                        print("[gateB] ABORT: too many errors", flush=True)
                        sys.exit(4)
            if seen % 50 == 0:
                el = time.time() - t0
                print(f"[gateB] {seen}/{total} cases {el / 60:.1f} min ETA {el / seen * (total - seen) / 60:.1f} min", flush=True)
        if args.job == "validity" and g == "anchor20":
            rws = {r["case_id"]: r for r in md.read_jsonl(str(rows_p)) if r["group"] == "anchor20"}
            agree = sum(int(np.argmax(r["direct"]["lp"]) == int(np.argmax(ga_rows[c]))) for c, r in rws.items())
            print(f"[gateB] anchor check vs Gate A: {agree}/{len(rws)} agree on direct argmax", flush=True)
            if len(rws) == min(20, len(cases)) and agree < min(19, len(rws)):
                json.dump({"anchor_agree": agree, "n": len(rws)}, open(run_dir / "ABORTED_ANCHOR.json", "w"))
                print("[gateB] ABORT: harness does not reproduce Gate A", flush=True)
                sys.exit(3)
    print("[gateB] finished all groups", flush=True)


if __name__ == "__main__":
    if args.selftest:
        selftest()
    else:
        run()
