#!/usr/bin/env python
"""A1 retrieval audit (docs/preregistration_v1_amendment_A1.md). Adapter ON (the frozen baseline pipeline), teacher-forced forward passes.
Stages inside one run: embed all train images with PubMedCLIP (cached) -> retrieve (R_test / R_full / random control) -> forward arms.
  python retrieval_audit.py [--run-dir DIR] [--limit N] | --selftest (CPU, no model)"""
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
CLIP = ROOT + "/models/pubmedclip_vit_b32"
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvqa_diag as md  # noqa: E402

cs = md.cs
SEED, K = 20261002, 3
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}
IDS = json.load(open(DER + "/ids.json"))
ROW = {c: i for i, c in enumerate(IDS)}
H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}
RTEST = set(json.load(open(POOLD + "/split_v1/r_test_ids.json")))
NT = set(json.load(open(POOLD + "/split_v1/n_test_ids.json")))
POOL = [c for c in IDS if c not in H]


def is_mcq(c):
    return T[c]["question type"] == "mcq"


def corpora():
    by_test, by_full = {}, {}
    for c in RTEST:
        if is_mcq(c):
            by_test.setdefault(T[c]["primary_modality"], []).append(c)
    for c in POOL:
        if is_mcq(c):
            by_full.setdefault(T[c]["primary_modality"], []).append(c)
    for d in (by_test, by_full):
        for k in d:
            d[k].sort()
    return by_test, by_full


def topk(E, qid, cands, k=K):
    rows = np.array([ROW[c] for c in cands])
    sims = E[rows] @ E[ROW[qid]]
    o = np.argsort(-sims, kind="stable")[:k]
    return [cands[i] for i in o], [float(sims[i]) for i in o]


def answer_text(cid):
    c = T[cid]
    return LAB.sub("", c[c["answer"].strip()]).strip()


def context_block(nids):
    lines = ["Reference cases with visually similar images (verified answers, for context):"]
    for n, cid in enumerate(nids, 1):
        q = " ".join(T[cid]["question"].split())[:300]
        lines.append(f"({n}) Q: {q}\n    Verified answer: {answer_text(cid)}")
    lines.append("")
    lines.append("Now the case to answer:")
    return "\n".join(lines)


def tau_from_val():
    rows = md.read_jsonl(glob.glob(ROOT + "/experiments/pilot/gateB_validity_s0_22436140_*/forward_rows.jsonl")[0])
    ms = []
    for r in rows:
        if r["group"] == "val700":
            p = sorted(md.softmax5(r["direct"]["lp"]), reverse=True)
            ms.append(p[0] - p[1])
    ms.sort()
    assert len(ms) == 700
    return ms[int(0.15 * len(ms))], ms  # 15th percentile (index 105 of 700)


def embed_all():
    path = DER + "/pubmedclip_img.npy"
    if os.path.exists(path):
        E = np.load(path)
        assert E.shape == (len(IDS), 512)
        return E
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP)
    out = []
    t0 = time.time()
    for s in range(0, len(IDS), 64):
        ims = [Image.open(POOLD + "/extracted/train/imgs/" + os.path.basename(T[c]["image_path"])).convert("RGB") for c in IDS[s:s + 64]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            v = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
        out.append(v.cpu().numpy())
        if (s // 64) % 40 == 0:
            print(f"[embed] {s}/{len(IDS)} {time.time() - t0:.0f}s", flush=True)
    E = np.concatenate(out).astype(np.float32)
    np.save(path, E)
    os.chmod(path, 0o444)
    del model
    torch.cuda.empty_cache()
    return E


def selftest():
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n, flush=True)
        if not ok:
            fails.append(n)
    bt, bf = corpora()
    nt_test = {c for d in bt.values() for c in d}
    nt_full = {c for d in bf.values() for c in d}
    check("R_test MCQ corpus excludes all test ids and N(T)", not (nt_test & H) and not (nt_test & NT))
    check("R_full MCQ corpus excludes test ids, includes some N(T)", not (nt_full & H) and bool(nt_full & NT))
    check("R_test is a subset of R_full", nt_test <= nt_full)
    check("every test MCQ modality has a corpus", all(T[c]["primary_modality"] in bt for c in H if is_mcq(c)))
    tau, ms = tau_from_val()
    check("tau_m from VAL700 15th percentile in (0,1)", 0.0 < tau < 1.0)
    print("     tau_m =", round(tau, 6), "| share of VAL700 below tau:", round(sum(m < tau for m in ms) / 700, 3), flush=True)
    rng = np.random.RandomState(0)
    E = rng.randn(len(IDS), 512).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    qid = sorted(c for c in H if is_mcq(c))[0]
    cand = bt[T[qid]["primary_modality"]]
    E[ROW[cand[5]]] = E[ROW[qid]]
    ids, sims = topk(E, qid, cand)
    check("topk returns the planted exact match first with sim ~1", ids[0] == cand[5] and abs(sims[0] - 1.0) < 1e-5 and len(ids) == K)
    blk = context_block(ids)
    check("context has 3 neighbours, no caption, no letter-only answers", blk.count("Verified answer:") == 3 and "visual_description" not in blk and all(T[i]["visual_description"][:40] not in blk for i in ids))
    check("answer text is the option text (no 'A) ' label)", not re.match(r"^[A-E]\)", answer_text(ids[0])))
    test = md.load_mcq_cases()
    check("711 test MCQ cases load", len(test) == 711)
    check("clip model dir and adapter dir present", os.path.exists(CLIP + "/config.json") and os.path.isdir(cs.ADAPTER_DIR))
    c2 = dataclasses.replace(test[0], question=blk + "\n\n" + test[0].question)
    check("context question keeps original question at the end", c2.question.endswith(test[0].question))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


def run():
    import torch
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"gateD_retrieval_s0_{jobid}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    tau, _ = tau_from_val()
    E = embed_all()
    bt, bf = corpora()
    test = md.load_mcq_cases()
    if args.limit:
        test = test[: args.limit]
    plan = {}
    for c in test:
        cid, mod = c.case_id, T[c.case_id]["primary_modality"]
        n1, s1 = topk(E, cid, bt[mod])
        n2, s2 = topk(E, cid, bf[mod])
        rnd = random.Random(f"{SEED}|{cid}").sample(bt[mod], K)
        assert not ({*n1, *n2, *rnd} & H) and not (set(n1) & NT) and not (set(rnd) & NT)
        plan[cid] = {"m1": n1, "m1_sims": s1, "m1leak": n2, "m1leak_sims": s2, "m1c": rnd, "modality": mod}
    if not resume:
        json.dump({"amendment": "A1", "script_sha256": md.sha256_file(__file__), "amendment_sha256": md.sha256_file(ROOT + "/docs/preregistration_v1_amendment_A1.md"),
                   "k": K, "tau_m": tau, "encoder": CLIP, "embeddings_sha256": md.sha256_file(DER + "/pubmedclip_img.npy"),
                   "split_r_test_sha256": md.sha256_file(POOLD + "/split_v1/r_test_ids.json"), "n_test": len(test)}, open(run_dir / "config.json", "w"), indent=2)
        json.dump(plan, open(run_dir / "retrieval_plan.json", "w"))
    system = cs.CustomMedReasonSystem(md.make_config())
    system.setup()
    torch_, device = system.torch, system.model.device
    assert hasattr(system.model, "peft_config"), "adapter must be ON"
    json.dump(md.env_info(system), open(run_dir / "env.json", "w"), indent=2)
    dids, cids = md.letter_token_ids(system.processor.tokenizer)
    ga = {}
    for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
        for r in md.read_jsonl(pth):
            if r["arm"] == "base":
                ga[r["case_id"]] = r["direct"]["lp"]
    rows_p = run_dir / "rows.jsonl"
    done = {(r["case_id"], r["arm"]) for r in md.read_jsonl(str(rows_p))}
    fh, eh = open(rows_p, "a"), open(run_dir / "errors.jsonl", "a")
    t0, n_err = time.time(), 0
    for ci, case in enumerate(test):
        cid = case.case_id
        arms = [("m1", "m1", False), ("m1c", "m1c", False), ("m1leak", "m1leak", False), ("m1black", "m1", True)]
        if ci < 20:
            arms = [("m0check", None, False)] + arms
        need = [a for a in arms if (cid, a[0]) not in done]
        if need:
            try:
                rgb = np.array(Image.open(case.primary_image_path).convert("RGB"))
                modality = system._detect(rgb)
                image = Image.fromarray(cs._preprocess(rgb, modality))
                for name, key, black in need:
                    img = Image.fromarray(np.zeros((224, 224, 3), np.uint8)) if black else image
                    c2 = case if key is None else dataclasses.replace(case, question=context_block(plan[cid][key]) + "\n\n" + case.question)
                    kw, n_direct, _ = md.prepare(system, c2, img, modality, device, torch_)
                    res = md.score_forward(system, kw, n_direct, dids, cids, torch_)
                    md.append_jsonl(fh, {"case_id": cid, "arm": name, "gold": T[cid]["answer"].strip(), "neighbours": plan[cid].get(key) if key else None,
                                         "n_direct_tokens": n_direct, **res})
                    del kw
            except Exception:  # noqa: BLE001
                n_err += 1
                md.append_jsonl(eh, {"case_id": cid, "traceback": traceback.format_exc()})
                print("[ret] ERROR", cid, traceback.format_exc(), flush=True)
                if ci == 0 or n_err > 10:
                    sys.exit(4)
        if ci == 19:
            rr = {r["case_id"]: r for r in md.read_jsonl(str(rows_p)) if r["arm"] == "m0check"}
            agree = sum(int(np.argmax(r["direct"]["lp"]) == int(np.argmax(ga[c]))) for c, r in rr.items())
            print(f"[ret] m0check vs Gate A: {agree}/{len(rr)} agree", flush=True)
            if len(rr) == 20 and agree < 19:
                json.dump({"agree": agree}, open(run_dir / "ABORTED_ANCHOR.json", "w"))
                sys.exit(3)
        if (ci + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[ret] {ci + 1}/{len(test)} {el / 60:.1f} min ETA {el / (ci + 1) * (len(test) - ci - 1) / 60:.1f} min", flush=True)
    print("[ret] finished", flush=True)


if __name__ == "__main__":
    selftest() if args.selftest else run()
