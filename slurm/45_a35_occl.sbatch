#!/bin/bash
# A35 occlusion evaluation of saved dose-response adapters. Rules: docs/preregistration_v1_amendment_A35.md (frozen before submission).
# Submit: sbatch -J a35_x --export=ALL,BENCH=pmc|aokvqa,ADAPTER=<run dir>,DOSE=0.25,LABEL=pmc025s11[,SMOKE=1] slurm/39_a35_eval.sbatch
#SBATCH --job-name=a35_e
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
export HF_HOME=${PROJECT_ROOT}/hf_cache TRANSFORMERS_CACHE=${PROJECT_ROOT}/hf_cache TMPDIR=${PROJECT_ROOT}/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=${PROJECT_ROOT}/tmp/triton_cache
mkdir -p "${HF_HOME}" "${TMPDIR}"
module load miniconda3/24.3.0
source activate /gscratch/gokulsrinathseetharam/conda_envs/medreason_env
echo "=== A35 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Label ${LABEL} Adapter ${ADAPTER} Smoke ${SMOKE:-0}"
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
python -u scripts/a35_occl_eval.py --bench "${BENCH}" --adapter "${ADAPTER}" --dose "${DOSE}" --label "${LABEL}" ${FLAG}
