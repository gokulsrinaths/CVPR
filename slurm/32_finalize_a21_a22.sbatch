#!/bin/bash
# Summarises every finished A21/A22 run that has no summary yet and runs the frozen analysis scripts. Idempotent; safe to resubmit. No result is recorded or hashed here.
#SBATCH --job-name=a2x_finalize
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
set -uo pipefail
cd /gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
PY=/gscratch/gokulsrinathseetharam/conda_envs/medreason_env/bin/python
for d in experiments/pilot/a21_*_20261* experiments/pilot/dose_*_seed20261012_2*; do
  case $d in *smoke*) continue;; esac
  [ -f $d/summary_dose.json ] && continue
  [ -f $d/rows_ft_orig.jsonl ] || continue
  case $d in experiments/pilot/a21_*) S=scripts/a21_summarize.py;; *) S=scripts/dose_summarize.py;; esac
  echo "summarising $d"; $PY $S $d > /dev/null 2>&1 || echo "FAILED $d"
done
echo "=== A21"; $PY scripts/a21_analysis.py | head -80
echo "=== A22"; $PY scripts/a22_analysis.py | head -120
echo FINALIZE_DONE
