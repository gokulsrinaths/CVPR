#!/bin/bash
# A22 runs on RTX PRO 6000 Blackwell GPUs (separate CUDA 12.8 environment). Rules: docs/preregistration_v1_amendment_A22.md (frozen before submission).
# Submit: sbatch -J a22_x --export=ALL,SCRIPT=a21_run.py|dose_run.py,FAMILY=q3b|q7b|ivl2b,SHORTCUT=length|position,DOSE=0.99,SEED=20261021,GC=0|1,ARMS=base+black+blur32+blur16+blur8+blur4+blur2 slurm/31_a22_r6000.sbatch
#SBATCH --job-name=a22_s
#SBATCH --account=effectorfold
#SBATCH --partition=mb-r6000
#SBATCH --gres=gpu:r6000:1
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
export PATH=/gscratch/gokulsrinathseetharam/conda_envs/medreason_bw/bin:$PATH
echo "=== A22 ===  Job ${SLURM_JOB_ID}  Node $(hostname)  Script ${SCRIPT} Family ${FAMILY} ${SHORTCUT} dose ${DOSE} seed ${SEED} GC ${GC:-0} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
python -c "import torch;print('torch',torch.__version__,torch.version.cuda)"
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
GCFLAG=""; if [ "${GC:-0}" = "1" ]; then GCFLAG="--gc"; fi
python -u scripts/${SCRIPT} --family "${FAMILY}" --shortcut "${SHORTCUT}" --dose "${DOSE}" --seed "${SEED}" --arms "${ARMS//+/,}" ${GCFLAG} ${FLAG}
