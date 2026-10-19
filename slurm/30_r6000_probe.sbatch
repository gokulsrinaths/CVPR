#!/bin/bash
#SBATCH --job-name=r6000_probe
#SBATCH --account=effectorfold
#SBATCH --partition=mb-r6000
#SBATCH --gres=gpu:r6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/logs/%x_%j.out
module load miniconda3/24.3.0
export PATH=/gscratch/gokulsrinathseetharam/conda_envs/medreason_bw/bin:$PATH
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
/gscratch/gokulsrinathseetharam/conda_envs/medreason_bw/bin/python - <<PY
import torch,time,transformers;print("arch",torch.cuda.get_arch_list()[-2:])
print("torch",torch.__version__,"cuda",torch.version.cuda,"bf16_supported",torch.cuda.is_bf16_supported(),"cap",torch.cuda.get_device_capability(0))
a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16);torch.cuda.synchronize();t=time.time()
for _ in range(50): b=a@a
torch.cuda.synchronize();print("bf16 matmul TFLOPS ~%.1f"%(50*2*4096**3/(time.time()-t)/1e12))
PY
