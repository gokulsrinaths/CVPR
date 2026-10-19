#!/usr/bin/env python
"""Same-source graph, step 1 (CPU): perceptual hashes (dHash, pHash; 64-bit) for every official train image.
Writes data/train_pool/derived/{ids.json,dhash.npy,phash.npy}. Images are read-only inputs; nothing else is touched."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
OUT = POOL + "/derived"
os.makedirs(OUT, exist_ok=True)

N = 32
k = np.arange(N)
DCT = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * N)) * np.sqrt(2.0 / N)
DCT[0] /= np.sqrt(2)


def bits_to_u64(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return np.uint64(v)


def hashes(path):
    im = Image.open(path).convert("L")
    d = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    dh = bits_to_u64((d[:, 1:] > d[:, :-1]).flatten())
    p = np.asarray(im.resize((N, N), Image.LANCZOS), dtype=np.float32)
    f = DCT @ p @ DCT.T
    low = f[:8, :8].flatten()
    ph = bits_to_u64(low > np.median(low[1:]))
    return dh, ph


def work(item):
    cid, p = item
    try:
        return cid, hashes(p)
    except Exception as e:  # noqa: BLE001
        return cid, str(e)


if __name__ == "__main__":
    T = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
    items = [(c["case_id"], POOL + "/extracted/train/imgs/" + os.path.basename(c["image_path"])) for c in T]
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        res = pool.map(work, items, chunksize=64)
    bad = [r for r in res if isinstance(r[1], str)]
    assert not bad, bad[:3]
    ids = [r[0] for r in res]
    np.save(OUT + "/dhash.npy", np.array([r[1][0] for r in res], dtype=np.uint64))
    np.save(OUT + "/phash.npy", np.array([r[1][1] for r in res], dtype=np.uint64))
    json.dump(ids, open(OUT + "/ids.json", "w"))
    print("DONE", len(ids), "images", round(time.time() - t0), "s", flush=True)
