#!/bin/bash
# Gate C / A4: Qwen2.5-VL-7B-Instruct image-dependence audit (second system). Rules: docs/preregistration_v1_amendment_A4.md (frozen before submission).
# 1,411 MCQs x 2 arms, greedy, max 8 new tokens, bf16. Estimate ~20-40 min on an H100.
# Submit: sbatch slurm/10_gateC_s2.sbatch      Resume: sbatch --export=ALL,RUN_DIR=<run dir> slurm/10_gateC_s2.sbatch
#SBATCH --job-name=gateC_s2_qwen25vl_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache



mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== gateC s2 (A4) ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
EXTRA=""; if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/audit_s2.py ${EXTRA}
