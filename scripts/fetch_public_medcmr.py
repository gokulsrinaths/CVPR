import os, sys, time
from huggingface_hub import hf_hub_download
t = time.time()
p = hf_hub_download(repo_id="aaassddaadf/Med-CMR", repo_type="dataset", filename="dataset.zip",
                    local_dir="/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/data/public_medcmr")
print("DONE", p, os.path.getsize(p), "bytes", round(time.time() - t), "s", flush=True)
