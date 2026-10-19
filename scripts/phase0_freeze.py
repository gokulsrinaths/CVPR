#!/usr/bin/env python
"""
Phase 0 -- freeze the existing MedReason/GA-CoT reproduction (job 13666563).

CPU-only. Copies (never moves) the frozen artifacts into
results/reproduction_frozen/, hashes everything, writes MANIFEST.json, and sets
the copies read-only. Refuses to run twice (never overwrites a frozen record).
Also creates the experiments/ layout and seeds experiments/job_registry.csv
from Slurm accounting for the four jobs already run.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
OUT = f"{ROOT}/results/reproduction_frozen"
ENV_PY = "/gscratch/gokulsrinathseetharam/conda_envs/medreason_env/bin/python"
JOB = "13666563"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    if os.path.exists(f"{OUT}/MANIFEST.json"):
        print("MANIFEST.json already exists -- refusing to overwrite a frozen record.")
        sys.exit(1)
    for d in (OUT, f"{ROOT}/analysis", f"{ROOT}/experiments/pilot", f"{ROOT}/experiments/confirmatory",
              f"{ROOT}/experiments/ablations"):
        os.makedirs(d, exist_ok=True)
    for sub in ("predictions", "logs", "code_and_slurm", "data_identity", "model_identity"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    H = f"{ROOT}/data/holdout886"
    copies = {
        "predictions/results.json": f"{H}/run_output/results.json",
        "predictions/scores.json": f"{H}/run_output/scores.json",
        "logs/gacot_real_holdout886_13666563.out": f"{ROOT}/logs/gacot_real_holdout886_13666563.out",
        "logs/gacot_real_holdout886_13666563.err": f"{ROOT}/logs/gacot_real_holdout886_13666563.err",
        "code_and_slurm/04_gacot_real_holdout886.sbatch": f"{ROOT}/slurm/04_gacot_real_holdout886.sbatch",
        "data_identity/cases.json": f"{H}/cases.json",
        "data_identity/ground_truth.json": f"{H}/ground_truth.json",
        "data_identity/holdout_886_case_ids.txt": f"{H}/holdout_886_case_ids.txt",
        "data_identity/SUMMARY.json": f"{H}/SUMMARY.json",
        "model_identity/adapter_config.json": f"{ROOT}/models/adapter/adapter_config.json",
        "model_identity/base_model_config.json": f"{ROOT}/models/base_model/config.json",
        "model_identity/base_model_index.json": f"{ROOT}/models/base_model/model.safetensors.index.json",
    }
    files = {}
    for rel, src in copies.items():
        dst = f"{OUT}/{rel}"
        shutil.copy2(src, dst)
        files[rel] = {"sha256": sha256_file(dst), "bytes": os.path.getsize(dst), "source": src}

    # code snapshot (deterministic order, no bytecode)
    code_tar = f"{OUT}/code_and_slurm/code_snapshot.tar"
    with tarfile.open(code_tar, "w") as t:
        for p in sorted(glob.glob(f"{ROOT}/code/**/*", recursive=True)):
            if os.path.isfile(p) and "__pycache__" not in p:
                t.add(p, arcname=os.path.relpath(p, ROOT), recursive=False)
    files["code_and_slurm/code_snapshot.tar"] = {"sha256": sha256_file(code_tar), "bytes": os.path.getsize(code_tar)}

    # model identity (heavy hashes of the small artifacts; base shards done below)
    ident = {}
    for name, p in (("adapter_model.safetensors", f"{ROOT}/models/adapter/adapter_model.safetensors"),
                    ("detector.pkl", f"{ROOT}/models/detector.pkl"),
                    ("adapter_training_args.bin", f"{ROOT}/models/adapter/training_args.bin")):
        ident[name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "path": p}
    acfg = json.load(open(f"{ROOT}/models/adapter/adapter_config.json"))
    bcfg = json.load(open(f"{ROOT}/models/base_model/config.json"))

    # environment
    pipfreeze = sh(f"{ENV_PY} -m pip freeze 2>/dev/null")
    open(f"{OUT}/code_and_slurm/medreason_env_pip_freeze.txt", "w").write(pipfreeze + "\n")
    pyver = sh(f"{ENV_PY} --version 2>&1")

    # slurm accounting for the four jobs already run
    jobs = ["12016890", "12031503", "12033676", JOB]
    sacct = {}
    for j in jobs:
        out = sh(f"sacct -j {j} -X -n -P --format=JobID,JobName%30,Partition,QOS,NodeList,AllocTRES%60,Elapsed,State,ExitCode,Submit,Start,End")
        sacct[j] = out
    open(f"{OUT}/logs/sacct_four_jobs.txt", "w").write("\n".join(sacct.values()) + "\n")

    manifest = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Frozen record of the MedReason 2026 GA-CoT reproduction (real 886-case held-out run).",
        "git_commit": None,
        "git_note": "Project is not a git repository; identity is pinned by sha256 of code_snapshot.tar instead.",
        "job": {"job_id": JOB, "sacct": sacct[JOB]},
        "model": {
            "base_model_hf_id": acfg.get("base_model_name_or_path"),
            "base_architectures": bcfg.get("architectures"),
            "base_model_type": bcfg.get("model_type"),
            "adapter": {"type": acfg.get("peft_type"), "r": acfg.get("r"), "lora_alpha": acfg.get("lora_alpha"),
                        "target_modules": acfg.get("target_modules"), "sha256": ident["adapter_model.safetensors"]["sha256"],
                        "bytes": ident["adapter_model.safetensors"]["bytes"]},
            "modality_detector_sha256": ident["detector.pkl"]["sha256"],
            "base_model_shards_sha256": "see base_model_shards_sha256.json (computed after manifest)",
        },
        "dataset": {
            "name": "MedReason 2026 seed=42 held-out slice",
            "n_cases": 886, "n_mcq": 711, "n_open": 175,
            "case_id_list_sha256_frozen": json.load(open(f"{H}/SUMMARY.json")).get("case_id_list_sha256"),
            "case_id_list_sha256_recomputed_file": files["data_identity/holdout_886_case_ids.txt"]["sha256"],
            "cases_json_sha256": files["data_identity/cases.json"]["sha256"],
            "ground_truth_json_sha256": files["data_identity/ground_truth.json"]["sha256"],
        },
        "decoding_and_pipeline (from code/medreason_docker/systems/custom_system.py, see code snapshot)": {
            "decoding": "greedy (do_sample=False); NO random seed is used or needed",
            "repetition_penalty": 1.05,
            "max_new_tokens_first_pass": {"mcq": 768, "open": 1024},
            "thinking": "enable_thinking=True; if '</think>' missing after first pass, forced cue "
                        "'</think> The correct option is' then 8 (mcq) / 256 (open) more tokens",
            "quantization": "bitsandbytes 4-bit NF4, double-quant, bf16 compute",
            "attention": "tries flash_attention_2 then silently falls back to sdpa inside try/except; "
                         "WHICH ONE RAN IS NOT LOGGED (reproducibility gap)",
            "unused_env_vars": "MEDREASON_TEMPERATURE and MEDREASON_MAX_NEW_TOKENS are read into RuntimeConfig "
                               "but not used by the custom system (token budgets are hard-coded)",
            "mcq_answer_extraction": "first standalone option letter after </think>; else last letter anywhere "
                                     "in raw text; else first option label (silent default)",
            "preprocessing": "image-statistics sklearn detector routes to standard / low_dose_xray / 7t_mri; "
                             "non-standard routes get OOD preprocessing + an acquisition note in the system prompt",
        },
        "evaluation": {
            "script": "scoring_suite/scoring/score_all.py -> medreason_scoring/scorer.py (MCQ branch)",
            "mcq_rule": "strict exact match on the option label: ans.strip() == gold.strip() (no case folding)",
            "mcq_result": {"n": 711, "correct": 683, "accuracy": 0.960619},
            "paper_reported": {"correct": 684, "n": 711, "accuracy_approx": 0.962},
            "open_ended": "proxy judges only -- scores.json records judges=proxy(NOT-OFFICIAL), official_metric=False. "
                          "NOT a reproduction of the paper's GT/VA numbers.",
        },
        "environment": {"python": pyver, "pip_freeze_file": "code_and_slurm/medreason_env_pip_freeze.txt",
                        "hardware_from_sacct": "1x H100 on mb-h100 (node mbh100-001)"},
        "files": files,
    }
    json.dump(manifest, open(f"{OUT}/MANIFEST.json", "w"), indent=2)

    # experiments registry
    reg = f"{ROOT}/experiments/job_registry.csv"
    hdr = ["job_id", "date", "method", "dataset", "split", "seed", "gpu", "runtime", "exit_code", "config", "output_path", "notes"]
    meta = {
        "12016890": ("h100_smoke_test", "none", "none", "synthetic", "slurm/01_h100_smoke_test.sbatch", "environment smoke test"),
        "12031503": ("gacot_mechanics_smoke", "synthetic_2case", "smoke", "none", "slurm/02_gacot_mechanics_smoke_test.sbatch", "2 synthetic cases; no accuracy claim"),
        "12033676": ("gacot_scoring_dryrun", "synthetic_20case", "smoke", "none", "slurm/03_gacot_scoring_dryrun.sbatch", "scoring pipeline dry run; proxy judges"),
        JOB: ("M0_gacot_baseline", "medreason_holdout886", "heldout", "none", "slurm/04_gacot_real_holdout886.sbatch", "MCQ 683/711; GT/VA proxy only"),
    }
    with open(reg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for j in jobs:
            cols = sacct[j].split("|")
            if len(cols) < 12:
                continue
            m = meta[j]
            w.writerow([j, cols[9][:10], m[0], m[1], m[2], m[3], "H100:1", cols[6], cols[8], m[4],
                        f"{ROOT}/results/reproduction_frozen/" if j == JOB else "", m[5]])
    print("manifest + registry written")

    # base-model shard hashes (slow, done last)
    shard = {}
    for p in sorted(glob.glob(f"{ROOT}/models/base_model/model.safetensors-*.safetensors")):
        shard[os.path.basename(p)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        print("hashed", os.path.basename(p), flush=True)
    json.dump(shard, open(f"{OUT}/model_identity/base_model_shards_sha256.json", "w"), indent=2)

    # make the frozen record read-only
    for root, _, fs in os.walk(OUT):
        for fn in fs:
            os.chmod(os.path.join(root, fn), 0o444)
    print("FROZEN:", OUT)


if __name__ == "__main__":
    main()
