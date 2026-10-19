#!/usr/bin/env python
"""
mvqa_diag -- Gate A' diagnostics on the 711 MCQ cases, reusing the frozen baseline pipeline UNMODIFIED.

Nothing in code/, models/, data/ or results/reproduction_frozen/ is touched. All output goes to a new
run directory under experiments/pilot/ (never overwritten; pass --run-dir to RESUME the same run).

Per case (in cases.json order) it runs, resumably:

  greedy_repeat  the unmodified CustomMedReasonSystem.predict_case (baseline path). processor.decode is wrapped
                 ONLY to record the raw generation. Gives: run-to-run / hardware determinism vs the frozen
                 predictions, the raw text that was never saved, and whether the forced-conclusion fallback fired.
  forward arms   one teacher-forced forward pass each, reading the answer-letter distribution:
                   base       identity (the baseline prompt)
                   black      image replaced by a black 224x224 image (the code's own missing-image fallback)
                   mismatch   image of a different case (fixed cyclic derangement, seed below)
                   perm1..4   options cyclically relabelled (content moves to a different letter)
                 Each forward pass reads TWO positions of ONE sequence:
                   direct = right after  "<stub>\n</think>\n\n"            (bare letter tokens)
                   cue    = after        "The correct option is"            (letter tokens with a leading space)
                 The stub is the model's own constant MCQ reasoning (see analysis/error_analysis_report.md).

Early-abort gate: after --gate-n cases, abort (exit 3) if greedy flips vs frozen or base-arm agreement with the
frozen predictions is out of bounds, so a broken configuration cannot burn GPU hours.

Modes:  (default) run | --selftest (CPU, no model) | --summarize RUN_DIR
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
FROZEN = f"{ROOT}/results/reproduction_frozen"
HOLDOUT = f"{ROOT}/data/holdout886"
os.environ.setdefault("MEDREASON_MODEL_PATH", f"{ROOT}/models/base_model")
os.environ.setdefault("MEDREASON_ADAPTER_PATH", f"{ROOT}/models/adapter")
os.environ.setdefault("MEDREASON_DETECTOR_PATH", f"{ROOT}/models/detector.pkl")
sys.path.insert(0, f"{ROOT}/code")

from medreason_docker.config import RuntimeConfig  # noqa: E402
from medreason_docker.io import load_cases  # noqa: E402
from medreason_docker.schema import MedReasonOption  # noqa: E402
from medreason_docker.systems import custom_system as cs  # noqa: E402

STUB = "I examine the image and identify the relevant findings."
LETTERS = "ABCDE"
SEED = 20260930
ARMS = ["base", "black", "mismatch", "perm1", "perm2", "perm3", "perm4"]


# --------------------------------------------------------------------------- pure helpers (unit-tested)
def rotate_case(case, s):
    """Relabel options: the text at label i becomes the ORIGINAL text at index (i - s) mod n."""
    opts = list(case.options)
    n = len(opts)
    assert n == 5 and [o.label for o in opts] == list(LETTERS), "expected labels A-E in order"
    new = tuple(MedReasonOption(label=opts[i].label, text=opts[(i - s) % n].text) for i in range(n))
    return dataclasses.replace(case, options=new)


def orig_index(j, s, n=5):
    """Model chose label index j under rotation s -> index of that content in the ORIGINAL option order."""
    return (j - s) % n


def softmax5(lp):
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    z = sum(e)
    return [x / z for x in e]


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties 0.5. labels: 1 = positive."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(fh, row):
    fh.write(json.dumps(row) + "\n")
    fh.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- prompt construction (mirrors predict_case)
def build_text(system, case, image, modality):
    sysp = cs.SYSTEM_PROMPT + cs.MAP_PROMPTS.get(modality, "")
    messages = [{"role": "system", "content": sysp},
                {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": system._user_text(case)}]}]
    return system.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True)


def prepare(system, case, image, modality, device, torch):
    text = build_text(system, case, image, modality)
    direct_text = text + STUB + "\n</think>\n\n"
    inputs = system.processor(text=[direct_text], images=[[image]], return_tensors="pt")
    ids = inputs["input_ids"]
    n_direct = int(ids.shape[1])
    cue_tok = system.processor.tokenizer("The correct option is", add_special_tokens=False,
                                         return_tensors="pt").input_ids
    cue_ids = torch.cat([ids, cue_tok], dim=1)
    kw = {"input_ids": cue_ids, "attention_mask": torch.ones_like(cue_ids)}
    for k in ("pixel_values", "image_grid_thw"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k]
    mm = inputs.get("mm_token_type_ids")  # required by Qwen3.5 M-RoPE; extend with 0 (text) over the cue tokens
    if mm is not None:
        kw["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros((mm.shape[0], cue_tok.shape[1]), dtype=mm.dtype)], dim=1)
    kw = {k: v.to(device) for k, v in kw.items()}
    return kw, n_direct, text


def letter_token_ids(tok):
    d, c = [], []
    for l in LETTERS:
        a = tok.encode(l, add_special_tokens=False)
        b = tok.encode(" " + l, add_special_tokens=False)
        assert len(a) == 1 and len(b) == 1, f"letter {l} is not a single token: {a} {b}"
        d.append(a[0])
        c.append(b[0])
    assert len(set(d)) == 5 and len(set(c)) == 5
    return d, c


def score_forward(system, kw, n_direct, dids, cids, torch):
    tok = system.processor.tokenizer
    with torch.inference_mode():
        out = system.model(**kw)
    logits = out.logits[0]
    res = {}
    for name, pos, ids in (("direct", n_direct - 1, dids), ("cue", logits.shape[0] - 1, cids)):
        lp = torch.log_softmax(logits[pos].float(), dim=-1)
        letters = [float(lp[i]) for i in ids]
        top_v, top_i = torch.topk(lp, 5)
        res[name] = {"lp": letters, "mass": float(sum(math.exp(x) for x in letters)),
                     "top5": [[tok.decode([int(i)]), float(v)] for v, i in zip(top_v, top_i)]}
    del out, logits
    return res


# --------------------------------------------------------------------------- summary
def summarize(run_dir):
    run_dir = Path(run_dir)
    gt = {g["case_id"]: g for g in json.load(open(f"{FROZEN}/data_identity/ground_truth.json"))["ground_truth"]}
    frozen = {a["case_id"]: a["answer"].strip() for a in json.load(open(f"{FROZEN}/predictions/results.json"))["answers"]}
    cases = {c.case_id: c for c in load_cases(Path(f"{FROZEN}/data_identity/cases.json"), input_dir=Path(HOLDOUT))}
    fwd = read_jsonl(run_dir / "forward_arms.jsonl")
    gre = read_jsonl(run_dir / "greedy_repeat.jsonl")
    out = {"run_dir": str(run_dir)}

    if gre:
        flips = [r["case_id"] for r in gre if r["parsed_answer"] != frozen[r["case_id"]]]
        acc = sum(r["parsed_answer"] == gt[r["case_id"]]["answer"].strip() for r in gre)
        out["greedy_repeat"] = {
            "n": len(gre), "correct": acc, "accuracy": round(acc / len(gre), 6),
            "flips_vs_frozen": len(flips), "flip_case_ids": flips,
            "forced_conclusion_path_used": sum(bool(r.get("forced_conclusion_path")) for r in gre),
            "raw_think_equals_stub": sum(STUB in (r.get("raw_decodes") or [""])[0] for r in gre),
        }

    by = {}
    for r in fwd:
        by.setdefault(r["arm"], {})[r["case_id"]] = r
    for variant in ("direct", "cue"):
        arms_out = {}
        for arm, rows in by.items():
            s = int(arm[4:]) if arm.startswith("perm") else 0
            correct = agree_frozen = n = 0
            mass = []
            for cid, r in rows.items():
                lp = r[variant]["lp"]
                j = max(range(5), key=lambda k: lp[k])
                pred_idx = orig_index(j, s)
                n += 1
                correct += int(pred_idx == LETTERS.index(gt[cid]["answer"].strip()))
                agree_frozen += int(LETTERS[pred_idx] == frozen[cid])
                mass.append(r[variant]["mass"])
            arms_out[arm] = {"n": n, "accuracy": round(correct / n, 4) if n else None,
                             "agreement_with_frozen_prediction": round(agree_frozen / n, 4) if n else None,
                             "mean_letter_mass": round(sum(mass) / len(mass), 4) if mass else None}
        out[f"arms_{variant}"] = arms_out

    # permutation self-consistency + confidence (direct variant; fall back to cue if direct mass is tiny)
    complete = [cid for cid in by.get("base", {}) if all(cid in by.get(a, {}) for a in ("base", "perm1", "perm2", "perm3", "perm4"))]
    for variant in ("direct", "cue"):
        if not complete:
            break
        fixed = broken = corr = 0
        agree_shortest = 0
        margins, corr_flags, err_margins, ok_margins = [], [], [], []
        for cid in complete:
            gold = LETTERS.index(gt[cid]["answer"].strip())
            prob = [0.0] * 5
            for arm in ("base", "perm1", "perm2", "perm3", "perm4"):
                s = int(arm[4:]) if arm.startswith("perm") else 0
                p = softmax5(by[arm][cid][variant]["lp"])
                for j in range(5):
                    prob[orig_index(j, s)] += p[j] / 5
            pick = max(range(5), key=lambda k: prob[k])
            base_lp = by["base"][cid][variant]["lp"]
            base_pick = max(range(5), key=lambda k: base_lp[k])
            corr += int(pick == gold)
            fixed += int(base_pick != gold and pick == gold)
            broken += int(base_pick == gold and pick != gold)
            opts = [o.text for o in cases[cid].options]
            agree_shortest += int(pick == min(range(5), key=lambda k: (len(opts[k]), k)))
            srt = sorted(softmax5(base_lp), reverse=True)
            m = srt[0] - srt[1]
            margins.append(m)
            corr_flags.append(int(base_pick == gold))
            (ok_margins if base_pick == gold else err_margins).append(m)
        med = lambda v: sorted(v)[len(v) // 2] if v else None
        out[f"permutation_self_consistency_{variant}"] = {
            "n": len(complete), "accuracy": round(corr / len(complete), 4),
            "fixed_vs_base": fixed, "broken_vs_base": broken,
            "picks_equal_to_shortest_option": round(agree_shortest / len(complete), 4),
            "base_prob_margin_auroc_for_correctness": auroc(margins, corr_flags),
            "base_margin_median_correct": med(ok_margins), "base_margin_median_errors": med(err_margins)}

    json.dump(out, open(run_dir / "summary.json", "w"), indent=2)
    return out


# --------------------------------------------------------------------------- run
def env_info(system):
    torch = system.torch
    info = {"host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    try:
        info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    for pkg in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    impl = {}
    m = system.model
    for label, getter in (("model.config", lambda: m.config),
                          ("base_model.config", lambda: m.get_base_model().config)):
        try:
            cfg = getter()
            impl[label] = getattr(cfg, "_attn_implementation", None)
            for sub in ("text_config", "vision_config"):
                if hasattr(cfg, sub):
                    impl[f"{label}.{sub}"] = getattr(getattr(cfg, sub), "_attn_implementation", None)
        except Exception as e:  # noqa: BLE001
            impl[label] = f"unavailable: {e}"
    info["attn_implementation_actually_loaded"] = impl
    return info


def make_config():
    return RuntimeConfig(input_dir=Path(HOLDOUT), output_dir=Path(tempfile.gettempdir()),
                         output_file=Path(tempfile.gettempdir()) / "unused.json", system_name="custom",
                         submission_name="mvqa_diag", model_path=os.environ["MEDREASON_MODEL_PATH"],
                         max_new_tokens=512, temperature=0.0)


def load_mcq_cases():
    cases = load_cases(Path(f"{FROZEN}/data_identity/cases.json"), input_dir=Path(HOLDOUT))
    return [c for c in cases if c.task_type == "mcq"]


def gate_check(run_dir, frozen, n, max_flips, min_agree, skip_greedy):
    fwd = [r for r in read_jsonl(run_dir / "forward_arms.jsonl") if r["arm"] == "base"][:n]
    gre = read_jsonl(run_dir / "greedy_repeat.jsonl")[:n]
    problems = []
    if not skip_greedy:
        flips = sum(r["parsed_answer"] != frozen[r["case_id"]] for r in gre)
        if flips > max_flips:
            problems.append(f"greedy flips vs frozen = {flips}/{len(gre)} > {max_flips}")
    if fwd:
        best = 0.0
        for variant in ("direct", "cue"):
            agree = sum(LETTERS[max(range(5), key=lambda k: r[variant]["lp"][k])] == frozen[r["case_id"]] for r in fwd) / len(fwd)
            best = max(best, agree)
        if best < min_agree:
            problems.append(f"base-arm agreement with frozen predictions = {best:.2f} < {min_agree}")
    return problems


def run(args):
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    ts = time.strftime("%Y%m%dT%H%M%S")
    resume = args.run_dir is not None
    run_dir = Path(args.run_dir) if resume else Path(ROOT) / "experiments" / "pilot" / f"{args.tag}_{jobid}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        json.dump({"args": vars(args), "script_sha256": sha256_file(__file__), "seed": SEED, "arms": ARMS,
                   "frozen_manifest_sha256": sha256_file(f"{FROZEN}/MANIFEST.json"), "stub": STUB,
                   "note": "greedy decoding; the only randomness is the fixed mismatch-image derangement."},
                  open(run_dir / "config.json", "w"), indent=2)
    print(f"[diag] run_dir={run_dir} resume={resume}", flush=True)

    system = cs.CustomMedReasonSystem(make_config())
    system.setup()
    torch = system.torch
    device = system.model.device
    json.dump(env_info(system), open(run_dir / "env.json", "w"), indent=2)
    print("[diag] env:", json.dumps(json.load(open(run_dir / "env.json"))), flush=True)
    dids, cids = letter_token_ids(system.processor.tokenizer)

    cases = load_mcq_cases()
    if args.limit:
        cases = cases[: args.limit]
    frozen = {a["case_id"]: a["answer"].strip() for a in json.load(open(f"{FROZEN}/predictions/results.json"))["answers"]}
    ids = [c.case_id for c in cases]
    k = random.Random(SEED).randrange(1, len(ids))
    donor = {ids[i]: ids[(i + k) % len(ids)] for i in range(len(ids))}
    by_id = {c.case_id: c for c in cases}

    done_f = {(r["case_id"], r["arm"]) for r in read_jsonl(run_dir / "forward_arms.jsonl")}
    done_g = {r["case_id"] for r in read_jsonl(run_dir / "greedy_repeat.jsonl")}
    fwd_fh = open(run_dir / "forward_arms.jsonl", "a")
    gre_fh = open(run_dir / "greedy_repeat.jsonl", "a")
    err_fh = open(run_dir / "errors.jsonl", "a")

    def load_rgb(case):
        p = case.primary_image_path
        return np.array(Image.open(p).convert("RGB")) if p and Path(p).exists() else np.zeros((224, 224, 3), np.uint8)

    t0 = time.time()
    n_err = 0
    gate_done = False
    for ci, case in enumerate(cases):
        need_arms = [a for a in ARMS if (case.case_id, a) not in done_f]
        need_greedy = (not args.skip_greedy) and case.case_id not in done_g
        if not need_arms and not need_greedy:
            continue
        rgb = load_rgb(case)
        modality = system._detect(rgb)
        image = Image.fromarray(cs._preprocess(rgb, modality))
        try:
            for arm in need_arms:
                ts0 = time.time()
                if arm == "base":
                    c2, img = case, image
                elif arm == "black":
                    c2, img = case, Image.fromarray(np.zeros((224, 224, 3), np.uint8))
                elif arm == "mismatch":
                    c2, img = case, Image.fromarray(cs._preprocess(load_rgb(by_id[donor[case.case_id]]), modality))
                else:
                    s = int(arm[4:])
                    c2, img = rotate_case(case, s), image
                kw, n_direct, _ = prepare(system, c2, img, modality, device, torch)
                res = score_forward(system, kw, n_direct, dids, cids, torch)
                append_jsonl(fwd_fh, {"case_id": case.case_id, "arm": arm, "gold_label": None,
                                      "modality_route": modality, "n_direct_tokens": n_direct,
                                      "seconds": round(time.time() - ts0, 3), **res})
                del kw
            if need_greedy:
                ts0 = time.time()
                captured = []
                orig = system.processor.decode

                def wrapped(*a, **kk):
                    s_ = orig(*a, **kk)
                    captured.append(s_)
                    return s_

                system.processor.decode = wrapped
                try:
                    pred = system.predict_case(case)
                finally:
                    system.processor.decode = orig
                append_jsonl(gre_fh, {"case_id": case.case_id, "parsed_answer": pred.answer.strip(),
                                      "frozen_answer": frozen[case.case_id],
                                      "flip": pred.answer.strip() != frozen[case.case_id],
                                      "reasoning_trace": pred.reasoning_trace, "raw_decodes": captured,
                                      "forced_conclusion_path": len(captured) > 1,
                                      "seconds": round(time.time() - ts0, 3)})
        except Exception:  # noqa: BLE001
            n_err += 1
            append_jsonl(err_fh, {"case_id": case.case_id, "traceback": traceback.format_exc()})
            print(f"[diag] ERROR on {case.case_id}:\n{traceback.format_exc()}", flush=True)
            if ci == 0 or n_err > max(5, 0.05 * len(cases)):
                print("[diag] aborting: failure on first case or >5% failures", flush=True)
                sys.exit(4)
        if (ci + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[diag] {ci + 1}/{len(cases)} cases, {el / 60:.1f} min elapsed, "
                  f"ETA {el / (ci + 1) * (len(cases) - ci - 1) / 60:.1f} min", flush=True)
        if not gate_done and (ci + 1) >= args.gate_n:
            gate_done = True
            problems = gate_check(run_dir, frozen, args.gate_n, args.gate_max_flips, args.gate_min_agree,
                                  args.skip_greedy)
            print(f"[diag] gate after {args.gate_n} cases: {'FAILED' if problems else 'passed'} {problems}", flush=True)
            if problems and not args.no_gate:
                json.dump({"gate_problems": problems}, open(run_dir / "ABORTED_GATE.json", "w"), indent=2)
                sys.exit(3)

    print("[diag] finished all cases; summarising", flush=True)
    print(json.dumps(summarize(run_dir), indent=2), flush=True)


# --------------------------------------------------------------------------- CPU self-test (no model, no GPU)
def selftest():
    import pickle

    import torch
    from transformers import AutoProcessor

    fails = []

    def check(name, cond, extra=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""), flush=True)
        if not cond:
            fails.append(name)

    cases = load_mcq_cases()
    c0 = cases[0]
    # 1 rotation mapping: content follows the rotation and maps back to the original index
    for s in (1, 2, 3, 4):
        rc = rotate_case(c0, s)
        ok = all(rc.options[j].text == c0.options[orig_index(j, s)].text for j in range(5))
        check(f"rotate_case shift={s} maps back to original index", ok)
        check(f"rotate_case shift={s} keeps labels A-E", [o.label for o in rc.options] == list(LETTERS))
    check("rotate_case is a permutation (all 5 texts present)", sorted(o.text for o in rotate_case(c0, 2).options) == sorted(o.text for o in c0.options))
    check("original case is not mutated (frozen dataclass)", [o.label for o in c0.options] == list(LETTERS))

    # 2 auroc / softmax
    check("auroc perfect", auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0)
    check("auroc tie = 0.5", abs(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-9)
    check("softmax5 sums to 1", abs(sum(softmax5([0.1, -2, 3, 0, 1])) - 1) < 1e-9)

    # 3 real processor: prompt ending, prefix property, letter tokens, arms differ as expected
    system = cs.CustomMedReasonSystem(make_config())
    system.processor = AutoProcessor.from_pretrained(os.environ["MEDREASON_MODEL_PATH"], trust_remote_code=True)
    system.detector = pickle.load(open(os.environ["MEDREASON_DETECTOR_PATH"], "rb"))
    tok = system.processor.tokenizer
    dids, cids = letter_token_ids(tok)
    check("letters are single distinct tokens (direct and ' X')", True, f"direct={dids} cue={cids}")
    rgb = np.array(Image.open(c0.primary_image_path).convert("RGB"))
    modality = system._detect(rgb)
    image = Image.fromarray(cs._preprocess(rgb, modality))
    text = build_text(system, c0, image, modality)
    print("TEMPLATE TAIL:", repr(text[-160:]), flush=True)
    check("chat template ends with an open <think> block", text.rstrip().endswith("<think>"))
    kw, n_direct, _ = prepare(system, c0, image, modality, "cpu", torch)
    ids = kw["input_ids"][0]
    direct_tail = tok.decode(ids[n_direct - 14:n_direct])
    print("DIRECT PREFIX TAIL:", repr(direct_tail), flush=True)
    check("direct prefix ends with </think> + blank line", direct_tail.endswith("</think>\n\n"))
    check("cue suffix tokens follow the direct prefix", tok.decode(ids[n_direct:]) == "The correct option is")
    check("pixel_values/image_grid_thw present", "pixel_values" in kw and "image_grid_thw" in kw)
    check("mm_token_type_ids present (required by the model's M-RoPE)", "mm_token_type_ids" in kw)
    if "mm_token_type_ids" in kw:
        mm = kw["mm_token_type_ids"]
        img_pad = tok.convert_tokens_to_ids("<|image_pad|>")
        n_pad = int((kw["input_ids"] == img_pad).sum())
        check("mm_token_type_ids has the same shape as input_ids", tuple(mm.shape) == tuple(kw["input_ids"].shape),
              f"{tuple(mm.shape)} vs {tuple(kw['input_ids'].shape)}")
        check("image tokens are flagged in mm_token_type_ids; cue tokens are text (0)",
              int((mm != 0).sum()) == n_pad and int(mm[0, n_direct:].sum()) == 0,
              f"flagged={int((mm != 0).sum())} image_pad_tokens={n_pad} cue_nonzero={int(mm[0, n_direct:].sum())}")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
    check("forward kwargs contain every key the model requires", required.issubset(set(kw.keys())), str(sorted(kw.keys())))
    kw2, n2, _ = prepare(system, rotate_case(c0, 1), image, modality, "cpu", torch)
    check("rotated options change the prompt tokens", kw2["input_ids"].shape != kw["input_ids"].shape or not torch.equal(kw2["input_ids"], kw["input_ids"]))
    blk = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
    kw3, n3, _ = prepare(system, c0, blk, modality, "cpu", torch)
    check("black image changes visual token count", n3 != n_direct, f"{n_direct} vs {n3}")

    # 4 resume + summary on synthetic rows
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        with open(p, "a") as fh:
            append_jsonl(fh, {"a": 1})
            append_jsonl(fh, {"a": 2})
        check("jsonl append/read round-trip", [r["a"] for r in read_jsonl(p)] == [1, 2])
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED", flush=True)
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mvqa_diag_holdout711_s0")
    ap.add_argument("--run-dir", default=None, help="resume an existing run directory")
    ap.add_argument("--limit", type=int, default=0, help="only the first N MCQ cases (debug)")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--gate-n", type=int, default=20)
    ap.add_argument("--gate-max-flips", type=int, default=3)
    ap.add_argument("--gate-min-agree", type=float, default=0.9)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", default=None)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
