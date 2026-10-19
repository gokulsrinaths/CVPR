#!/bin/bash
# A30 UL on Qwen2.5-VL-7B on Med-CMR. Rules: docs/preregistration_v1_amendment_A30.md (frozen before submission).
# Submit: sbatch -J a30_x --export=ALL,SCRIPT=ft_run10.py,METHOD=ul|lpul|both,SEED=20261041[,SMOKE=1] slurm/42_a30_train.sbatch
#SBATCH --job-name=a30_s
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A30 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Script ${SCRIPT:-ft_run10.py} Method ${METHOD} Seed ${SEED} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
python -u scripts/${SCRIPT:-ft_run10.py} --model 7b --gc --mode satw --method "${METHOD}" --seed "${SEED}" ${FLAG}
