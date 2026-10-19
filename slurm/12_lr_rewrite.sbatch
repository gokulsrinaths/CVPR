#!/bin/bash
# A6 step 1: length-equalising rewrite of the options (Qwen2.5-VL-7B text-only rewriter, label-blind). Rules: docs/preregistration_v1_amendment_A6.md (frozen before submission).
# 1,411 questions, batches of 12, up to 2 attempts. Estimate ~30-60 min on an H100.
# Submit: sbatch slurm/12_lr_rewrite.sbatch      Resume: sbatch --export=ALL,RUN_DIR=<run dir> slurm/12_lr_rewrite.sbatch
#SBATCH --job-name=lr_rewrite_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache



mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A6 rewrite ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
EXTRA=""; if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/lr_rewrite.py ${EXTRA}
