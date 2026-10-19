# Note to amendment A22: hardware assignment of the runs (written before any A22 run was submitted)

Written 2026-10-04 after the two smoke tests of A22. A22 states that all runs use the RTX PRO 6000 Blackwell GPUs (partition mb-r6000). Only one of the twelve cards was free (the others were held by other users), and the heavy runs would have run one after another for about 12 hours. Therefore:
- Part H (2 runs) and Part P (3 runs), Qwen2.5-VL-3B on A-OKVQA, run on the RTX PRO 6000 with the CUDA 12.8 environment, as stated in A22.
- Part S (6 runs, second seed of A19: InternVL3-2B and Qwen2.5-VL-7B on PMC-VQA) run on A30 GPUs (partition beartooth-gpu) in the shared environment (PyTorch 2.7.1, CUDA 12.6) with gradient checkpointing, exactly as the A19 runs, so that the second seed uses the same hardware and software as the first.
No rule, dose, seed, script or analysis changes. The smoke test of Qwen-7B on the RTX PRO 6000 was cancelled while pending and is not needed. The smoke test of the position shortcut on the RTX PRO 6000 (job 23256833) completed without error; its outputs are test output only.
