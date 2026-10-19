#!/bin/bash
# A13 planted length cue on PMC-VQA. Rules: docs/preregistration_v1_amendment_A13.md (frozen before submission).
# Submit: sbatch -J a13_q3b --export=ALL,MODEL_KEY=q3b slurm/21_a13.sbatch   (smoke: add SMOKE=1)
#SBATCH --job-name=a13_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model
export MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter
export MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A13 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Model ${MODEL_KEY} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
python -u scripts/pmc_run2.py --model "${MODEL_KEY}" ${FLAG}
