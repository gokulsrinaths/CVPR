#!/usr/bin/env python
"""
Metadata-only inspection of the public Med-CMR zip via HTTP range requests (reads the central directory and the
first bytes of small metadata members). Does NOT download the 6.5 GB archive and builds no index.
"""
import collections
import json
import os
import re
import zipfile

from huggingface_hub import HfFileSystem

REPO = "datasets/aaassddaadf/Med-CMR/dataset.zip"
fs = HfFileSystem()
f = fs.open(REPO, "rb", block_size=4 * 1024 * 1024)
z = zipfile.ZipFile(f)
infos = z.infolist()
print("entries:", len(infos), "| archive bytes:", fs.size(REPO))

ext = collections.Counter(os.path.splitext(i.filename)[1].lower() for i in infos if not i.is_dir())
print("extensions:", dict(ext.most_common(8)))
tops = collections.Counter(i.filename.split("/")[0] for i in infos)
print("top-level names:", dict(tops.most_common(12)))
dirs2 = collections.Counter("/".join(i.filename.split("/")[:2]) for i in infos if not i.is_dir())
print("second-level groups:", dict(dirs2.most_common(12)))

meta = [i for i in infos if not i.is_dir() and os.path.splitext(i.filename)[1].lower() in (".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".parquet")]
print("metadata-like members:", [(i.filename, i.file_size) for i in meta][:25])

imgs = [i.filename for i in infos if i.filename.lower().endswith(".png")]
pat = collections.Counter(re.sub(r"\d+", "N", os.path.basename(p)) for p in imgs)
print("image filename patterns:", dict(pat.most_common(8)))
print("sample image names:", imgs[:5], "...", imgs[-3:])
kinds = collections.Counter()
for p in imgs:
    b = os.path.basename(p)
    m = re.match(r"^(Med-CMR-[a-z]+)-(\d+)", b)
    if m:
        kinds[m.group(1)] += 1
print("images by kind:", dict(kinds))

for i in meta[:6]:
    if i.file_size > 60_000_000:
        print("skip large member", i.filename, i.file_size)
        continue
    with z.open(i) as m:
        head = m.read(2500).decode("utf-8", errors="replace")
    print("\n=== head of", i.filename, f"({i.file_size} bytes) ===\n", head[:1800])
