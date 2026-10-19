#!/bin/bash
# A37: judge scoring and frozen analysis. Rules: docs/preregistration_v1_amendment_A37.md (frozen before submission).
#SBATCH --job-name=a37_judge
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err
set -euo pipefail
PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
cd "${PROJECT_ROOT}"
echo "=== A37 judge ===  Job ${SLURM_JOB_ID}  Node $(hostname)"
python -u scripts/a37_judge.py
echo A37_JUDGE_DONE
