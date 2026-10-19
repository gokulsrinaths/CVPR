# medical-vqa-retrieve-verify — Slurm setup

Cluster: Medicine Bow (ARCC, University of Wyoming)
Login node: mblog2 (via `ssh medicinebow`)
Account: effectorfold
Partition: mb-h100
QOS: fast
GPU: 1x H100 80GB

Conda env: `/gscratch/gokulsrinathseetharam/conda_envs/verifier_env` (PyTorch 2.6.0+cu124, already installed — reused for this project's smoke test).

## Submit the smoke test

```bash
cd /gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify
sbatch slurm/01_h100_smoke_test.sbatch
```

## Monitor

```bash
squeue -u gokulsrinathseetharam
```

## Cancel

```bash
scancel JOB_ID
```

## Logs

Output and error logs land in `logs/h100_smoke_test_<jobid>.out` and `.err`.
