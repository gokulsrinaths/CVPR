#!/bin/bash
# A44: Qwen2.5-VL-3B fine-tuned on the first N questions of the A9 training order. Rules: docs/preregistration_v1_amendment_A44.md (frozen before submission).
# Submit: sbatch -J a44_n1000_s3 --export=ALL,NTRAIN=1000,SEED=20261003[,SMOKE=1] slurm/51_a44_subset.sbatch
#SBATCH --job-name=a44_sub
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A44 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  N ${NTRAIN}  Seed ${SEED}  Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
python -u scripts/a44_ft_subset.py --model 3b --n-train "${NTRAIN}" --seed "${SEED}" ${FLAG}
