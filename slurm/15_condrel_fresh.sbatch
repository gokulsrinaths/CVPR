#!/bin/bash
# A8: confirmatory test on 3,000 fresh MCQs (arms base, black, mismatch_rand, mismatch_nn), adapter ON. Rules: docs/preregistration_v1_amendment_A8.md (frozen before submission). Rules: docs/preregistration_v1_amendment_A6.md (frozen before submission).
# 3,000 cases x 4 arms (+20 guard), teacher-forced. Estimate ~40-70 min on an H100.
# Submit: sbatch slurm/15_condrel_fresh.sbatch      Resume: sbatch --export=ALL,RUN_DIR=<run dir> slurm/15_condrel_fresh.sbatch
#SBATCH --job-name=condrel_fresh_s0
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
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
echo "=== A8 condrel ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
EXTRA=""; if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/cond_reliance.py ${EXTRA}
