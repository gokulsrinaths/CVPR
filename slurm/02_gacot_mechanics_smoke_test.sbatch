#!/bin/bash
#SBATCH --job-name=gacot_mechanics_smoke
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify

export HF_HOME=${PROJECT_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache
export TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${HF_HOME}" "${TMPDIR}"

export MEDREASON_SYSTEM=custom
export MEDREASON_INPUT_DIR=${PROJECT_ROOT}/data/smoke_test_input
export MEDREASON_OUTPUT_DIR=${PROJECT_ROOT}/data/smoke_test_output
export MEDREASON_OUTPUT_FILE=${MEDREASON_OUTPUT_DIR}/results.json
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model
export MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter
export MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
export TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache

module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env

echo "=== MECHANICS-ONLY SMOKE TEST ==="
echo "This run uses synthetic placeholder images, NOT official MedReason held-out data."
echo "It validates that the model/adapter/detector/code pipeline runs end-to-end on H100."
echo "It does NOT produce a valid benchmark score."
echo
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo

cd "${PROJECT_ROOT}/code"
python process.py

echo
echo "=== RESULTS ==="
cat "${MEDREASON_OUTPUT_FILE}"
