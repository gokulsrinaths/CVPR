#!/usr/bin/env python
"""Read-only peek at a RUNNING mvqa_diag run: copies the jsonl rows to a scratch dir and summarises the copy."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/scripts")
import mvqa_diag as md  # noqa: E402

rd = Path(sys.argv[1])
rows = md.read_jsonl(rd / "forward_arms.jsonl")
first = rows[0]["case_id"]
print("FIRST CASE", first)
for r in rows:
    if r["case_id"] != first:
        break
    lp = r["direct"]["lp"]
    j = max(range(5), key=lambda k: lp[k])
    top3 = [(t, round(v, 2)) for t, v in r["direct"]["top5"][:3]]
    print("  %-9s sec=%5.2f ntok=%4d direct_argmax=%s mass=%.3f cue_mass=%.3f top3=%s" % (
        r["arm"], r["seconds"], r["n_direct_tokens"], "ABCDE"[j], r["direct"]["mass"], r["cue"]["mass"], top3))
g = md.read_jsonl(rd / "greedy_repeat.jsonl")
if g:
    x = g[0]
    print("GREEDY", x["case_id"], "parsed=", x["parsed_answer"], "frozen=", x["frozen_answer"], "flip=", x["flip"],
          "forced_path=", x["forced_conclusion_path"], "raw0=", repr(x["raw_decodes"][0][:100]))

with tempfile.TemporaryDirectory(dir="/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/tmp") as d:
    for fn in ("forward_arms.jsonl", "greedy_repeat.jsonl"):
        shutil.copy(rd / fn, Path(d) / fn)
    s = md.summarize(d)
print("PARTIAL SUMMARY (n so far)")
print(json.dumps({k: v for k, v in s.items() if k != "run_dir"}, indent=1)[:3800])
