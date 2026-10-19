#!/bin/bash
#SBATCH --job-name=gacot_scoring_dryrun
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
SUITE=${PROJECT_ROOT}/scoring_suite/scoring

export HF_HOME=${PROJECT_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache
export TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${HF_HOME}" "${TMPDIR}"

export MEDREASON_SYSTEM=custom
export MEDREASON_INPUT_DIR=${SUITE}/heldout
export MEDREASON_OUTPUT_DIR=${SUITE}/run_output
export MEDREASON_OUTPUT_FILE=${MEDREASON_OUTPUT_DIR}/results.json
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model
export MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter
export MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
export TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
mkdir -p "${MEDREASON_OUTPUT_DIR}"

module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env

echo "=== SCORING-SUITE DRY RUN (synthetic heldout scaffold, proxy judges) ==="
echo "cases.json here is a structural scaffold with placeholder images, NOT the"
echo "organizers' real held-out set. This validates the scoring pipeline only."
echo
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo

echo "--- Step 1: inference over heldout/cases.json (20 cases) ---"
cd "${PROJECT_ROOT}/code"
python process.py

echo
echo "--- Step 2: validate_submission.py ---"
cd "${SUITE}"
python validate_submission.py --cases heldout/cases.json --results "${MEDREASON_OUTPUT_FILE}"

echo
echo "--- Step 3: score_all.py (proxy judges) ---"
python score_all.py --cases heldout/cases.json --ground-truth heldout/ground_truth.json \
    --results "${MEDREASON_OUTPUT_FILE}" --out "${MEDREASON_OUTPUT_DIR}/scores.json"

echo
echo "=== SCORES ==="
cat "${MEDREASON_OUTPUT_DIR}/scores.json"
