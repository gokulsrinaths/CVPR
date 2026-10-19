#!/bin/bash
# A38: judge-side counterfactuals with Qwen2.5-VL judge JUDGE = 3b|7b|32b. Rules: docs/preregistration_v1_amendment_A38.md (frozen before submission).
# Submit: sbatch -J a38_7b --export=ALL,JUDGE=7b slurm/48_a38_judge.sbatch  (32b: add -p mb-h100 --gres=gpu:h100:1 --qos=fast --mem=128G)
#SBATCH --job-name=a38_judge
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
cd "${PROJECT_ROOT}"
echo "=== A38 judge ${JUDGE} ===  Job ${SLURM_JOB_ID}  Node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -u scripts/a38_judge.py --judge "${JUDGE}"
echo A38_DONE
