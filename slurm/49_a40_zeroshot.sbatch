#!/bin/bash
# A40: zero-shot Qwen2.5-VL on six benchmarks with true / black / told arms. Rules: docs/preregistration_v1_amendment_A40.md (frozen before submission).
# Submit: sbatch -J a40_7b --export=ALL,MODEL=7b slurm/49_a40_zeroshot.sbatch   (32b: add -p mb-h100 --gres=gpu:h100:1 --qos=fast --mem=128G)
#SBATCH --job-name=a40_zs
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${PROJECT_ROOT}/data/aokvqa/pylibs
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
echo "=== A40 ${MODEL} ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Smoke ${SMOKE:-0}"
python -u scripts/a40_zeroshot.py --model "${MODEL}" ${FLAG}
echo A40_DONE
