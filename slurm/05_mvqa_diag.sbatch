#!/bin/bash
# Gate A' diagnostics (mvqa_diag): greedy repeat + teacher-forced answer-letter distributions on the 711 MCQ cases.
# Reuses the frozen baseline pipeline unmodified; writes only under experiments/pilot/<run-id>/.
#
# Submit:            sbatch slurm/05_mvqa_diag.sbatch
# Resume a run:      sbatch --export=ALL,RUN_DIR=<existing run dir> slurm/05_mvqa_diag.sbatch
# Any-GPU fallback:  sbatch --partition=mb-h100,mb-a30 --gres=gpu:1 --time=05:00:00 slurm/05_mvqa_diag.sbatch
#                    (the 4-bit 9B model fits on an A30; expect ~2-3x slower; results record the GPU used)
#
# Resources: 4 CPUs (this cluster appears to bill CPUs, not GPUs or memory, for fair-share), 64G RAM as OOM
# insurance for the 4-bit weight load (the baseline used 64G and worked), 1 GPU, 2.5 h (estimate ~1-1.3 h on H100).
#SBATCH --job-name=mvqa_diag_holdout711_s0
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
export HF_HOME=${PROJECT_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache
export TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
export MEDREASON_MODEL_PATH=${PROJECT_ROOT}/models/base_model
export MEDREASON_ADAPTER_PATH=${PROJECT_ROOT}/models/adapter
export MEDREASON_DETECTOR_PATH=${PROJECT_ROOT}/models/detector.pkl
mkdir -p "${HF_HOME}" "${TMPDIR}"

module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env

echo "=== mvqa_diag (Gate A') ==="
echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)   Partition: ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
echo

cd "${PROJECT_ROOT}"
EXTRA=""
if [ -n "${RUN_DIR:-}" ]; then EXTRA="--run-dir ${RUN_DIR}"; fi
python -u scripts/mvqa_diag.py --tag mvqa_diag_holdout711_s0 ${EXTRA}
