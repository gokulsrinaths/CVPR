#!/bin/bash
#SBATCH --job-name=h100_smoke_test
#SBATCH --account=effectorfold
#SBATCH --partition=mb-h100
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
#SBATCH --error=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify

export HF_HOME=${PROJECT_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache
export TMPDIR=${PROJECT_ROOT}/tmp
mkdir -p "${HF_HOME}" "${TMPDIR}"

module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/verifier_env

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo

python "${PROJECT_ROOT}/scripts/check_environment.py"
