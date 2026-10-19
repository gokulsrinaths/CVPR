#!/bin/bash
# A20 exploratory tempered rescue. Rules: docs/preregistration_v1_amendment_A20.md (frozen before submission).
# Submit: sbatch -J a15_x --export=ALL,KIND=pmc|medcmr,MODEL_KEY=q3b|3b,ADAPTER=run_dir,COND=orig|implant,ARMS=a,b slurm/23_a15.sbatch
#SBATCH --job-name=a20_s
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
echo "=== A18 ===  Job ${SLURM_JOB_ID}  Kind ${KIND} Mode ${MODE:-} Dose ${DOSE:-} Seed ${SEED} Fold ${FOLD:-} Smoke ${SMOKE:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cd "${PROJECT_ROOT}"
FLAG=""; if [ "${SMOKE:-0}" = "1" ]; then FLAG="--smoke"; fi
GCFLAG=""; if [ "${GC:-0}" = "1" ]; then GCFLAG="--gc"; fi
if [ "${KIND}" = "medpoe" ]; then
  python -u scripts/ft_run8.py --model 3b --mode satw --seed "${SEED}" --bias-dirs "${BIAS//+/,}" --bias-temp "${TEMP}" ${GCFLAG} ${FLAG}
else
  EXTRA=""; [ -n "${FOLD:-}" ] && EXTRA="$EXTRA --fold ${FOLD}"; [ -n "${BIAS:-}" ] && EXTRA="$EXTRA --bias-dirs ${BIAS//+/,}"
  python -u scripts/rescue_run.py --family q3b --shortcut length --dose "${DOSE}" --seed "${SEED}" --mode "${MODE}" ${EXTRA} --arms "${ARMS//+/,}" ${GCFLAG} ${FLAG}
fi
