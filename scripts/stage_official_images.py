#!/usr/bin/env python
"""Extract ONLY the official train + validation images (20,254) from the public Med-CMR zip into data/train_pool/extracted/.
The 400 reserved leaderboard images are never extracted. Every file is re-hashed and compared with public_image_hashes.json.
Writes data/train_pool/IMAGE_MANIFEST.json (name -> sha256, per split) and makes the staged tree read-only."""
import hashlib, json, os, stat, time, zipfile

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
t0 = time.time()
tr = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
va = json.load(open(POOL + "/official_json/medreason_validation_participant_facing.json", encoding="utf-8"))["cases"]
pub_hash = json.load(open(ROOT + "/data/public_medcmr/public_image_hashes.json"))
z = zipfile.ZipFile(ROOT + "/data/public_medcmr/dataset.zip")
members = {os.path.basename(i.filename): i for i in z.infolist() if i.filename.startswith("dataset/imgs/") and i.filename.lower().endswith(".png")}
manifest, bad = {}, []
for split, cases in (("train", tr), ("val", va)):
    d = POOL + "/extracted/" + split + "/imgs"
    os.makedirs(d, exist_ok=True)
    m = {}
    for c in cases:
        name = os.path.basename(c["image_path"])
        dst = d + "/" + name
        h = hashlib.sha256()
        with z.open(members[name]) as f, open(dst, "wb") as g:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
                g.write(b)
        if h.hexdigest() != pub_hash[name]:
            bad.append(name)
        m[name] = h.hexdigest()
    manifest[split] = m
    print(split, len(m), flush=True)
json.dump(manifest, open(POOL + "/IMAGE_MANIFEST.json", "w"))
os.chmod(POOL + "/IMAGE_MANIFEST.json", 0o444)
for dp, dns, fns in os.walk(POOL + "/extracted"):
    for fn in fns:
        os.chmod(os.path.join(dp, fn), 0o444)
    os.chmod(dp, 0o555)
print("DONE train=%d val=%d hash_mismatch=%d reserved_extracted=0 time=%ds" % (len(manifest["train"]), len(manifest["val"]), len(bad), time.time() - t0), flush=True)
