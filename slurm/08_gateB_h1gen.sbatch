#!/bin/bash
# Gate B / H1' (amendment A3): base model, no adapter, generated answer with thinking off. Rules: docs/preregistration_v1_amendment_A3.md (frozen before submission).
# 711 test MCQs x 2 arms, greedy, max 16 new tokens. Estimate ~15-30 min on an H100.
# Submit: sbatch slurm/08_gateB_h1gen.sbatch      Resume: sbatch --export=ALL,RUN_DIR=<run dir> slurm/08_gateB_h1gen.sbatch
#SBATCH --job-name=gateB_h1gen_base_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model
export MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter_NOT_USED
export MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== gateB h1gen (A3) ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
EXTRA=""; if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/audit_generate.py ${EXTRA}
