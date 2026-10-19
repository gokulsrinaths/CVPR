#!/bin/bash
# A24 interventions on the real Med-CMR shortcut. Rules: docs/preregistration_v1_amendment_A24.md (frozen before submission).
# Submit: sbatch -J a24_x --export=ALL,METHOD=lenprior|mu|both,SEED=20261031[,SMOKE=1][,GC=1] slurm/34_a24_a30.sbatch
#SBATCH --job-name=a24_s
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
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A24 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Method ${METHOD} Seed ${SEED} GC ${GC:-0} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
GCFLAG=""; if [ "${GC:-0}" = "1" ]; then GCFLAG="--gc"; fi
python -u scripts/ft_run9.py --model 3b --mode satw --method "${METHOD}" --seed "${SEED}" ${GCFLAG} ${FLAG}
