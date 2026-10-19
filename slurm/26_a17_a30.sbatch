#!/bin/bash
# A17 shortcut dose-response. Rules: docs/preregistration_v1_amendment_A17.md (frozen before submission).
# Submit: sbatch -J a15_x --export=ALL,KIND=pmc|medcmr,MODEL_KEY=q3b|3b,ADAPTER=run_dir,COND=orig|implant,ARMS=a,b slurm/23_a15.sbatch
#SBATCH --job-name=a17_s
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
echo "=== A17 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Family ${FAMILY} ${SHORTCUT} dose ${DOSE} seed ${SEED} GC ${GC:-0} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
GCFLAG=""; if [ "${GC:-0}" = "1" ]; then GCFLAG="--gc"; fi
python -u scripts/dose_run.py --family "${FAMILY}" --shortcut "${SHORTCUT}" --dose "${DOSE}" --seed "${SEED}" --arms "${ARMS//+/,}" ${GCFLAG} ${FLAG}
