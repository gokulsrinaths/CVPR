#!/usr/bin/env python
"""Same-source graph + dev split v1 (CPU only), per docs/dev_split_thresholds_rule.md and analysis/threshold_review.json.
Edges: E0 identical image sha256, E1 identical normalised caption, E3 caption Jaccard >= tau_cap (0.9). E2 not used (tau_img=0). E4: none found.
Writes data/train_pool/split_v1/* (read-only) and analysis/split_v1_summary.json."""
import collections
import hashlib
import json
import os

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
OUT = POOL + "/split_v1"
assert not os.path.exists(OUT), "split_v1 already exists; never overwrite"
os.makedirs(OUT)
SEED = "20260930"
T = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
ids = [c["case_id"] for c in T]
idx = {c: i for i, c in enumerate(ids)}
P = json.load(open(POOL + "/derived/pairs.json"))
tau_cap = json.load(open(ROOT + "/analysis/threshold_review.json"))["chosen"]["tau_cap"]
H = {l.strip() for l in open(ROOT + "/results/reproduction_frozen/data_identity/holdout_886_case_ids.txt") if l.strip()}

par = list(range(len(ids)))


def find(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        par[max(ra, rb)] = min(ra, rb)


edges = {"E0": [], "E1": [], "E3": []}
for grp in P["E0_groups"]:
    for j in grp[1:]:
        edges["E0"].append((grp[0], j))
for grp in P["E1_groups"]:
    for j in grp[1:]:
        edges["E1"].append((grp[0], j))
for i, j, jc in P["E3_pairs"]:
    if jc >= tau_cap:
        edges["E3"].append((i, j))
for et, es in edges.items():
    for a, b in es:
        union(a, b)
comp = collections.defaultdict(list)
for i in range(len(ids)):
    comp[find(i)].append(i)
comp_id = {}
for r, mem in comp.items():
    cid = min(ids[m] for m in mem)
    for m in mem:
        comp_id[ids[m]] = cid
size = {cid: 0 for cid in set(comp_id.values())}
for c in comp_id.values():
    size[c] += 1

test_idx = {idx[c] for c in H}
pool = [c for c in ids if c not in H]
test_comps = {comp_id[c] for c in H}
N_T = sorted(c for c in pool if comp_id[c] in test_comps)
R_test = sorted(c for c in pool if comp_id[c] not in test_comps)

# per-edge-type test cases that have a partner in the pool (adjacency by that edge type only)
adj = {et: collections.defaultdict(set) for et in edges}
for et, es in edges.items():
    for a, b in es:
        adj[et][a].add(b)
        adj[et][b].add(a)
# E0/E1 groups are connected through the group head; use group membership for exact partner counts
grp_of = {"E0": {}, "E1": {}}
for et, key in (("E0", "E0_groups"), ("E1", "E1_groups")):
    for gi, grp in enumerate(P[key]):
        for m in grp:
            grp_of[et][m] = gi
rows = {"E0": P["E0_groups"], "E1": P["E1_groups"]}
partner = {}
for et in ("E0", "E1"):
    cnt = 0
    for t in test_idx:
        if t in grp_of[et]:
            members = rows[et][grp_of[et][t]]
            if any(ids[m] not in H for m in members):
                cnt += 1
    partner[et] = cnt
partner["E3"] = sum(1 for t in test_idx if any(ids[b] not in H for b in adj["E3"].get(t, ())))
partner["any_edge_via_component"] = sum(1 for c in H if any(comp_id[p] == comp_id[c] for p in pool))

# ---- dev split: whole components containing no test case, stratified by question type x primary modality
strata = collections.defaultdict(list)  # stratum -> list of component ids (assigned by smallest member)
byid = {c["case_id"]: c for c in T}
pool_comps = collections.defaultdict(list)
for c in R_test:
    pool_comps[comp_id[c]].append(c)
stratum_total = collections.Counter()
for c in R_test:
    stratum_total[(byid[c]["question type"], byid[c]["primary_modality"])] += 1
for cid in pool_comps:
    strata[(byid[cid]["question type"], byid[cid]["primary_modality"])].append(cid)
D = []
for st, cids in sorted(strata.items()):
    target = 0.15 * stratum_total[st]
    got = 0
    for cid in sorted(cids, key=lambda c: hashlib.sha256((SEED + "|" + c).encode()).hexdigest()):
        if got >= target:
            break
        D.extend(pool_comps[cid])
        got += len(pool_comps[cid])
D = sorted(D)
Dset = set(D)
R_dev = sorted(c for c in R_test if c not in Dset)
big = sorted(((s, cid) for cid, s in size.items() if s > 50), reverse=True)

assert not (Dset & set(N_T)) and not (Dset & H) and not (set(R_dev) & Dset)
assert all(comp_id[c] not in {comp_id[x] for x in R_dev} for c in D), "a component spans D and R_dev"
assert len(Dset) + len(R_dev) + len(N_T) == len(pool) == 16836

summ = {"tau_cap": tau_cap, "edges_used": {k: len(v) for k, v in edges.items()}, "n_components": len(size),
        "component_size_hist": dict(sorted(collections.Counter(size.values()).items())),
        "components_over_50": big, "pool": len(pool), "N_T_size": len(N_T), "R_test_size": len(R_test),
        "D_size": len(D), "R_dev_size": len(R_dev),
        "D_mcq": sum(byid[c]["question type"] == "mcq" for c in D), "D_open": sum(byid[c]["question type"] == "open-ended" for c in D),
        "test_cases_with_partner_in_pool_by_edge": partner,
        "test_cases_total": len(H), "test_mcq_with_component_partner": sum(1 for c in H if c.startswith("MCQ_") and any(comp_id[p] == comp_id[c] for p in pool)),
        "N_T_by_question_type": dict(collections.Counter(byid[c]["question type"] for c in N_T)),
        "D_share_of_pool": round(len(D) / len(pool), 4)}
for name, obj in (("dev_ids", D), ("r_dev_ids", R_dev), ("r_test_ids", R_test), ("n_test_ids", N_T), ("test_ids", sorted(H))):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"))
json.dump(comp_id, open(OUT + "/components.json", "w"))
json.dump(summ, open(OUT + "/split_summary.json", "w"), indent=1)
json.dump(summ, open(ROOT + "/analysis/split_v1_summary.json", "w"), indent=1)
with open(OUT + "/SHA256.txt", "w") as f:
    for fn in sorted(os.listdir(OUT)):
        if fn != "SHA256.txt":
            f.write(hashlib.sha256(open(f"{OUT}/{fn}", "rb").read()).hexdigest() + "  " + fn + "\n")
for fn in os.listdir(OUT):
    os.chmod(f"{OUT}/{fn}", 0o444)
os.chmod(OUT, 0o555)
print(json.dumps(summ, indent=1))
