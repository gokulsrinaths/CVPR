#!/bin/bash
# Gate D / A1: retrieval audit (adapter ON): PubMedCLIP embeddings, M1 M1c M1leak M1black on the 711 test MCQs. Rules: docs/preregistration_v1_amendment_A1.md (frozen before submission).
# Embeds 17,722 train images (~5-10 min, cached) then 4 teacher-forced arms x 711 cases (+20 guard). Estimate ~25-40 min on an H100.
# Submit: sbatch slurm/11_gateD_retrieval.sbatch      Resume: sbatch --export=ALL,RUN_DIR=<run dir> slurm/11_gateD_retrieval.sbatch
#SBATCH --job-name=gateD_retrieval_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
echo "=== gateD retrieval (A1) ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
EXTRA=""; if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/retrieval_audit.py ${EXTRA}
