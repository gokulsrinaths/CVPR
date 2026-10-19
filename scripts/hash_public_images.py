import hashlib, json, os, time, zipfile
ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
t = time.time()
z = zipfile.ZipFile(ROOT + "/data/public_medcmr/dataset.zip")
d = {}
for info in z.infolist():
    n = info.filename
    if n.startswith("dataset/imgs/") and n.lower().endswith(".png"):
        h = hashlib.sha256()
        with z.open(info) as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        d[os.path.basename(n)] = h.hexdigest()
json.dump(d, open(ROOT + "/data/public_medcmr/public_image_hashes.json", "w"))
print("DONE", len(d), "images", round(time.time() - t), "s", flush=True)
