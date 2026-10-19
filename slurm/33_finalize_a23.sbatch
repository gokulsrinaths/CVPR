#!/bin/bash
# Summarises finished A23 runs and runs the frozen analysis. Idempotent; records/hashes nothing.
#SBATCH --job-name=a23_finalize
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
cd /gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
PY=/gscratch/gokulsrinathseetharam/conda_envs/medreason_env/bin/python
for d in experiments/pilot/a21_q3b_length0.9[79]_seed20261022_2* experiments/pilot/a21_q7b_length*_seed20261021_2*; do
  case $d in *smoke*) continue;; esac
  [ -f $d/summary_dose.json ] && continue
  [ -f $d/rows_ft_orig.jsonl ] || continue
  $PY scripts/a21_summarize.py $d > /dev/null 2>&1 || echo "FAILED $d"
done
$PY scripts/a23_analysis.py
echo FINALIZE_DONE
