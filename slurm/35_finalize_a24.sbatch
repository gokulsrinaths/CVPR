#!/bin/bash
# Summarises finished A24 runs (a12_summarize.py m2) and runs the frozen analysis. Idempotent; records/hashes nothing.
#SBATCH --job-name=a24_finalize
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
cd /gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
PY=/gscratch/gokulsrinathseetharam/conda_envs/medreason_env/bin/python
for d in experiments/pilot/a24_medcmr_*_3b_seed2026103[123]_2*; do
  case $d in *smoke*) continue;; esac
  [ -f $d/summary_a12.json ] && continue
  [ -f $d/rows_ft.jsonl ] || continue
  $PY scripts/a12_summarize.py m2 $d > /dev/null 2>&1 || echo "FAILED $d"
done
$PY scripts/a24_analysis.py
echo FINALIZE_DONE
