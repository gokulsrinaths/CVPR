#!/bin/bash
# Runs the frozen A27 analysis. Idempotent; records/hashes nothing.
#SBATCH --job-name=a27_finalize
#SBATCH --account=effectorfold
#SBATCH --partition=beartooth
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
cd /gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
/gscratch/gokulsrinathseetharam/conda_envs/medreason_env/bin/python scripts/a27_analysis.py
echo FINALIZE_DONE
