import platform
import shutil
import subprocess
import sys

import torch


def main():
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA version (PyTorch build): {torch.version.cuda}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"GPU count visible: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i}: {name} ({total_mem_gb:.1f} GB)")
    else:
        print("WARNING: No CUDA device visible to PyTorch.")

    print("-" * 60)
    print("nvidia-smi output:")
    print("-" * 60)
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print(smi.stdout)
    except Exception as e:
        print(f"nvidia-smi failed: {e}")

    if cuda_available:
        print("-" * 60)
        print("GPU tensor multiplication test:")
        print("-" * 60)
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"Result shape: {tuple(c.shape)}, dtype: {c.dtype}, device: {c.device}")
        print(f"Sample value c[0,0]: {c[0, 0].item():.4f}")
        print("GPU tensor multiplication: SUCCESS")

    print("-" * 60)
    print("Scratch disk space:")
    print("-" * 60)
    usage = shutil.disk_usage("/gscratch/gokulsrinathseetharam")
    print(f"Total: {usage.total / (1024 ** 3):.1f} GB")
    print(f"Used:  {usage.used / (1024 ** 3):.1f} GB")
    print(f"Free:  {usage.free / (1024 ** 3):.1f} GB")

    print("=" * 60)
    print("ENVIRONMENT CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
