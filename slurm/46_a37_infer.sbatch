#!/bin/bash
# A37: released adapter on the 175 open-ended held-out cases with image arm ARM = true|black|wrong. Rules: docs/preregistration_v1_amendment_A37.md (frozen before submission).
# Submit: sbatch -J a37_black --export=ALL,ARM=black slurm/46_a37_infer.sbatch
#SBATCH --job-name=a37_infer
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
IN=${PROJECT_ROOT}/data/holdout886_open_${ARM}
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
export MEDREASON_SYSTEM=custom MEDREASON_INPUT_DIR=${IN} MEDREASON_OUTPUT_DIR=${IN}/run_output MEDREASON_OUTPUT_FILE=${IN}/run_output/results.json
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
mkdir -p "${HF_HOME}" "${TMPDIR}" "${MEDREASON_OUTPUT_DIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A37 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Arm ${ARM}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}/code"
python process.py
echo A37_INFER_DONE
