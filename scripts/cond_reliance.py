#!/usr/bin/env python
"""A8: confirmatory test on FRESH3000. Adapter ON; arms base, black, mismatch_rand, mismatch_nn; teacher-forced direct+cue.
python cond_reliance.py [--run-dir DIR] [--limit N] | --selftest"""
import argparse
import dataclasses
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
ap.add_argument("--run-dir", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
DER = POOLD + "/derived"
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvqa_diag as md  # noqa: E402
from medreason_docker.schema import MedReasonCase, MedReasonOption  # noqa: E402

cs = md.cs
SEED, N = 20261003, 3000
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}
IDS = json.load(open(DER + "/ids.json"))
ROW = {c: i for i, c in enumerate(IDS)}
E = np.load(DER + "/pubmedclip_img.npy")
RTEST = set(json.load(open(POOLD + "/split_v1/r_test_ids.json")))
COMP = json.load(open(POOLD + "/split_v1/components.json"))
H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}


def excluded_ids():
    dev = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/groups.json")[0]))["dev700"]
    pool = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*/groups.json")[0]))["pool700"]
    return set(dev), set(pool)


def sample_ids():
    dev, pool = excluded_ids()
    cand = sorted(c for c in RTEST if T[c]["question type"] == "mcq" and c not in dev and c not in pool)
    return sorted(random.Random(SEED).sample(cand, N)), cand


def img_path(cid):
    return POOLD + "/extracted/train/imgs/" + os.path.basename(T[cid]["image_path"])


def mk(cid):
    opts = tuple(MedReasonOption(label=l, text=LAB.sub("", T[cid][l]).strip()) for l in md.LETTERS)
    return MedReasonCase(case_id=cid, task_type="mcq", question=T[cid]["question"].strip(), image_paths=(Path(img_path(cid)),), options=opts, metadata={})


def donors(ids):
    k = random.Random(SEED).randrange(1, len(ids))
    rand = {ids[i]: ids[(i + k) % len(ids)] for i in range(len(ids))}
    by_mod = {}
    for c in RTEST:
        if T[c]["question type"] == "mcq":
            by_mod.setdefault(T[c]["primary_modality"], []).append(c)
    for m in by_mod:
        by_mod[m].sort()
    members = {}
    for c, cc in COMP.items():
        members.setdefault(cc, set()).add(c)
    nn, sims = {}, {}
    for c in ids:
        bad = members[COMP[c]] | {c}
        cand = [x for x in by_mod[T[c]["primary_modality"]] if x not in bad]
        rows = np.array([ROW[x] for x in cand])
        s = E[rows] @ E[ROW[c]]
        j = int(np.argmax(s))
        nn[c], sims[c] = cand[j], float(s[j])
    return rand, nn, sims


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    ids, cand = sample_ids()
    ids2, _ = sample_ids()
    dev, pool = excluded_ids()
    check("sample is deterministic, 3000 unique ids", ids == ids2 and len(set(ids)) == N)
    check("sample disjoint from DEV700, POOL700, the 886 test ids", not (set(ids) & dev) and not (set(ids) & pool) and not (set(ids) & H))
    check("sample is MCQ-only and inside R_test", all(T[c]["question type"] == "mcq" and c in RTEST for c in ids))
    print("     eligible candidates:", len(cand), flush=True)
    check("eligible pool is at least 3x the sample", len(cand) >= 3 * N)
    rand, nn, sims = donors(ids)
    check("random donors form a derangement", all(rand[c] != c for c in ids) and len(set(rand.values())) == N)
    check("nn donors are never the case itself or a same-source case", all(nn[c] != c and COMP[nn[c]] != COMP[c] for c in ids))
    check("nn donors are same-modality and not test cases", all(T[nn[c]]["primary_modality"] == T[c]["primary_modality"] and nn[c] not in H for c in ids))
    check("images exist for cases and donors", all(os.path.exists(img_path(x)) for c in ids[:300] for x in (c, rand[c], nn[c])))
    print("     mean nn cosine similarity:", round(sum(sims.values()) / len(sims), 4), "| min:", round(min(sims.values()), 4), flush=True)
    check("embeddings shape", E.shape == (len(IDS), 512))
    check("adapter directory present (adapter ON)", os.path.isdir(cs.ADAPTER_DIR))
    j1 = glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*/forward_rows.jsonl")
    check("J1 anchor rows exist", bool(j1))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


def run():
    ids, _ = sample_ids()
    rand, nn, sims = donors(ids)
    if args.limit:
        ids = ids[: args.limit]
    j1_run = glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*")[0]
    j1_ids = json.load(open(j1_run + "/groups.json"))["pool700"][:20]
    j1_lp = {r["case_id"]: r["direct"]["lp"] for r in md.read_jsonl(j1_run + "/forward_rows.jsonl") if r["group"] == "pool700"}
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"condrel_fresh_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        json.dump({"amendment": "A8", "script_sha256": md.sha256_file(__file__), "amendment_sha256": md.sha256_file(ROOT + "/docs/preregistration_v1_amendment_A8.md"),
                   "mvqa_diag_sha256": md.sha256_file(ROOT + "/scripts/mvqa_diag.py"), "embeddings_sha256": md.sha256_file(DER + "/pubmedclip_img.npy"),
                   "seed": SEED, "n": N, "sample_ids": ids, "nn_donors": nn, "nn_sims": sims, "rand_donors": rand}, open(run_dir / "config.json", "w"))
    system = cs.CustomMedReasonSystem(md.make_config())
    system.setup()
    torch, device = system.torch, system.model.device
    assert hasattr(system.model, "peft_config"), "adapter must be ON"
    json.dump(md.env_info(system), open(run_dir / "env.json", "w"), indent=2)
    dids, cids = md.letter_token_ids(system.processor.tokenizer)
    rows_p = run_dir / "rows.jsonl"
    done = {(r["group"], r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
    fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")

    def rgb_of(cid):
        return np.array(Image.open(img_path(cid)).convert("RGB"))

    t0, n_err, seen = time.time(), 0, 0
    plan = [("anchor", [(c, ["base"]) for c in j1_ids])] + [("fresh", [(c, ["base", "black", "mismatch_rand", "mismatch_nn"]) for c in ids])]
    total = sum(len(v) for _, v in plan)
    for g, items in plan:
        for cid, arms in items:
            seen += 1
            need = [a for a in arms if (g, cid, a) not in done]
            if need:
                try:
                    case = mk(cid)
                    rgb = rgb_of(cid)
                    modality = system._detect(rgb)
                    image = Image.fromarray(cs._preprocess(rgb, modality))
                    for arm in need:
                        if arm == "base":
                            img = image
                        elif arm == "black":
                            img = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                        else:
                            d = rand[cid] if arm == "mismatch_rand" else nn[cid]
                            img = Image.fromarray(cs._preprocess(rgb_of(d), modality))
                        kw, n_direct, _ = md.prepare(system, case, img, modality, device, torch)
                        res = md.score_forward(system, kw, n_direct, dids, cids, torch)
                        md.append_jsonl(fh, {"group": g, "case_id": cid, "arm": arm, "gold": T[cid]["answer"].strip(), "modality_route": modality, **res})
                        del kw
                except Exception:  # noqa: BLE001
                    n_err += 1
                    md.append_jsonl(eh, {"case_id": cid, "traceback": traceback.format_exc()})
                    print("[cond] ERROR", cid, traceback.format_exc(), flush=True)
                    if seen == 1 or n_err > 15:
                        sys.exit(4)
            if g == "anchor" and cid == j1_ids[-1]:
                rr = {r["case_id"]: r for r in md.read_jsonl(str(rows_p)) if r["group"] == "anchor"}
                agree = sum(int(np.argmax(r["direct"]["lp"]) == int(np.argmax(j1_lp[c]))) for c, r in rr.items())
                print(f"[cond] anchor vs J1: {agree}/{len(rr)} agree", flush=True)
                if len(rr) == 20 and agree < 19:
                    json.dump({"agree": agree}, open(run_dir / "ABORTED_ANCHOR.json", "w"))
                    sys.exit(3)
            if seen % 100 == 0:
                el = time.time() - t0
                print(f"[cond] {seen}/{total} {el / 60:.1f} min ETA {el / seen * (total - seen) / 60:.1f} min", flush=True)
    print("[cond] finished", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    selftest() if args.selftest else run()
