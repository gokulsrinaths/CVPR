import time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", local_dir="/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/models/qwen25vl_7b_instruct", allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","LICENSE*"])
print("DONE", p, round(time.time()-t), "s", flush=True)
