#!/usr/bin/env python
"""A12 analyses. python a12_summarize.py l1 RUN_DIR | l2 RUN_DIR | m2 RUN_DIR | weights TXA_DIR TXB_DIR OUT.json"""
import glob
import json
import math
import os
import re
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import gold_from_train  # noqa: E402
from len_rand import pad_options  # noqa: E402

LET = "ABCDE"
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = gold_from_train()


def read_jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def softmax(l):
    m = max(l)
    e = [math.exp(x - m) for x in l]
    s = sum(e)
    return [x / s for x in e]


def margin(l):
    p = sorted(softmax(l), reverse=True)
    return p[0] - p[1]


def pred(l):
    return LET[max(range(5), key=lambda k: l[k])]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n))


def table(rows, group):
    d = {}
    for r in rows:
        if r["group"] == group:
            d.setdefault(r["arm"], {})[r["case_id"]] = r["direct"]["lp"]
    return d


def opts(cid):
    return [LAB.sub("", T[cid][l]).strip() for l in LET]


def short_letter(cid, cond):
    o = pad_options(cid, opts(cid), cond) if cond != "orig" else opts(cid)
    L = [len(x) for x in o]
    m = min(L)
    return LET[L.index(m)] if L.count(m) == 1 else None


def final_rank_of_gold(cid, cond):
    o = pad_options(cid, opts(cid), cond) if cond != "orig" else opts(cid)
    L = [len(x) for x in o]
    g = LET.index(T[cid]["answer"].strip())
    return 1 + sum(L[i] < L[g] for i in range(5))


def acc(t, ids, arm="base"):
    return sum(pred(t[arm][i]) == T[i]["answer"].strip() for i in ids) / len(ids)


def eff(t_true, t_black, ids):
    gold = {i: T[i]["answer"].strip() for i in ids}
    ct = {i: pred(t_true[i]) == gold[i] for i in ids}
    cb = {i: pred(t_black[i]) == gold[i] for i in ids}
    fx = sum(ct[i] and not cb[i] for i in ids)
    br = sum((not ct[i]) and cb[i] for i in ids)
    m = len(ids)
    return {"n": m, "acc_true": round(sum(ct.values()) / m, 4), "acc_black": round(sum(cb.values()) / m, 4), "diff_points": round(100 * (sum(ct.values()) - sum(cb.values())) / m, 2),
            "fixes": fx, "breaks": br, "mcnemar_p": mcnemar_exact(fx, br) if fx + br else None}


def sig(e, thr):
    return bool(e["n"] and e["diff_points"] >= thr and e["mcnemar_p"] is not None and e["mcnemar_p"] < 0.05)


def a9_runs():
    return {"3b": glob.glob(ROOT + "/experiments/pilot/ft_3b_s0_22569048_*")[0], "7b": glob.glob(ROOT + "/experiments/pilot/ft_7b_s0_22569049_*")[0]}


mode = sys.argv[1]
if mode == "weights":
    rows = read_jsonl(os.path.join(sys.argv[2], "rows_tx_score.jsonl")) + read_jsonl(os.path.join(sys.argv[3], "rows_tx_score.jsonl"))
    m = {r["case_id"]: margin(r["direct"]["lp"]) for r in rows if r["group"] == "trfold"}
    w = {i: (1.0 if v < 0.90 else 0.25) for i, v in m.items()}
    json.dump(w, open(sys.argv[4], "w"))
    ms = sorted(m.values())
    print({"n": len(w), "share_undecided_lt0.90": round(sum(v < 0.90 for v in m.values()) / len(m), 4), "median_margin": round(ms[len(ms) // 2], 5)})
    sys.exit(0)

run_dir = sys.argv[2]
cfg = json.load(open(os.path.join(run_dir, "config.json")))
out = {"run_dir": run_dir, "mode": mode, "adapter_in": cfg.get("adapter_in")}
if mode == "l1":
    key = "3b" if "qwen25vl_3b" in cfg["model"] else "7b"
    a9 = table(read_jsonl(os.path.join(a9_runs()[key], "rows_ft.jsonl")), "fresh")
    a9t = table(read_jsonl(os.path.join(a9_runs()[key], "rows_ft.jsonl")), "test711")
    res = {}
    for cond in ("orig", "pres", "rand"):
        rows = read_jsonl(os.path.join(run_dir, f"rows_l1_{cond}.jsonl"))
        r = {}
        for g in ("test711", "fresh"):
            t = table(rows, g)
            if not t:
                continue
            ids = sorted(set(t["base"]) & set(t["black"]))
            d = {"n": len(ids), "acc_true": round(acc(t, ids), 4), "acc_black": round(acc(t, ids, "black"), 4)}
            sh = {i: short_letter(i, cond) for i in ids}
            d["pred_equals_final_shortest"] = round(sum(pred(t["base"][i]) == sh[i] for i in ids) / len(ids), 4)
            os_ = {i: short_letter(i, "orig") for i in ids}
            d["pred_equals_original_shortest"] = round(sum(pred(t["base"][i]) == os_[i] for i in ids) / len(ids), 4)
            by = {}
            for i in ids:
                by.setdefault(final_rank_of_gold(i, cond), []).append(pred(t["base"][i]) == T[i]["answer"].strip())
            d["acc_by_final_rank_of_correct"] = {str(k): [len(v), round(sum(v) / len(v), 4)] for k, v in sorted(by.items())}
            r[g] = d
        res[cond] = r
    out["conditions"] = res
    ref_test = round(acc(a9t, sorted(a9t["base"])), 4)
    ref_fresh = round(acc(a9, sorted(a9["base"])), 4)
    out["a9_reference_acc"] = {"test711": ref_test, "fresh": ref_fresh}
    v = {}
    v["V_orig_matches_a9"] = abs(res["orig"]["test711"]["acc_true"] - ref_test) <= 0.005
    v["V_filler_harmless_test"] = abs(res["pres"]["test711"]["acc_true"] - ref_test) <= 0.03
    v["V_filler_harmless_fresh"] = abs(res["pres"]["fresh"]["acc_true"] - ref_fresh) <= 0.03
    v["L1a_drop_ge20_test"] = (res["pres"]["test711"]["acc_true"] - res["rand"]["test711"]["acc_true"]) >= 0.20
    v["L1a_drop_ge20_fresh"] = (res["pres"]["fresh"]["acc_true"] - res["rand"]["fresh"]["acc_true"]) >= 0.20
    v["L1b_pred_final_shortest_ge50_test"] = res["rand"]["test711"]["pred_equals_final_shortest"] >= 0.50
    v["L1b_pred_final_shortest_ge50_fresh"] = res["rand"]["fresh"]["pred_equals_final_shortest"] >= 0.50
    out["verdicts"] = v
elif mode == "l2":
    rows = read_jsonl(os.path.join(run_dir, "rows_l2.jsonl"))
    tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    out["training"] = {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]}
    val = table(rows, "val700")["base"]
    ms = sorted(margin(l) for l in val.values())
    tau = ms[int(0.15 * len(ms))]
    out["tau_m_model"] = tau
    te = table(rows, "test711")
    ids = sorted(set(te["base"]) & set(te["black"]))
    out["TEST711_rand"] = {"acc_true": round(acc(te, ids), 4), "acc_black": round(acc(te, ids, "black"), 4),
                           "pred_equals_final_shortest": round(sum(pred(te["base"][i]) == short_letter(i, "rand") for i in ids) / len(ids), 4),
                           "pred_equals_original_shortest": round(sum(pred(te["base"][i]) == short_letter(i, "orig") for i in ids) / len(ids), 4)}
    f = table(rows, "fresh")
    ids = sorted(set(f["base"]) & set(f["black"]))
    mb = {i: margin(f["black"][i]) for i in ids}
    strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids}
    S = {s: [i for i in ids if strat[i] == s] for s in ("HI", "MID", "LO")}
    chg = lambda sub: sum(pred(f["base"][i]) != pred(f["black"][i]) for i in sub)
    out["FRESH_rand"] = {"n": len(ids), "acc_true": round(acc(f, ids), 4), "acc_black": round(acc(f, ids, "black"), 4), "all": eff(f["base"], f["black"], ids),
                         "share_black_margin_ge_0.99": round(sum(mb[i] >= 0.99 for i in ids) / len(ids), 4), "strata_sizes": {s: len(v) for s, v in S.items()},
                         "by_stratum": {s: eff(f["base"], f["black"], v) for s, v in S.items() if v}, "HI_changed": chg(S["HI"]), "HI_n": len(S["HI"]),
                         "pred_equals_final_shortest": round(sum(pred(f["base"][i]) == short_letter(i, "rand") for i in ids) / len(ids), 4),
                         "pred_equals_original_shortest": round(sum(pred(f["base"][i]) == short_letter(i, "orig") for i in ids) / len(ids), 4)}
    v = {"L2a_habit_gone": out["TEST711_rand"]["pred_equals_final_shortest"] <= 0.35,
         "L2b_text_decisive_le50": out["FRESH_rand"]["share_black_margin_ge_0.99"] <= 0.50,
         "L2c_gap_ge5": sig(out["FRESH_rand"]["all"], 5.0)}
    out["verdicts"] = v
elif mode == "m2":
    rows = read_jsonl(os.path.join(run_dir, "rows_ft.jsonl"))
    tl = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    out["training"] = {"first_loss": tl[0]["mean_loss"], "last_loss": tl[-1]["mean_loss"], "minutes": tl[-1]["minutes"]}
    a9r = read_jsonl(os.path.join(a9_runs()["3b"], "rows_ft.jsonl"))
    a9f = table(a9r, "fresh")
    a9val = sorted(margin(l) for l in table(a9r, "val700")["base"].values())
    a9tau = a9val[int(0.15 * len(a9val))]
    ids = sorted(set(a9f["base"]) & set(a9f["black"]))
    lo_ids = [i for i in ids if margin(a9f["black"][i]) < 0.90]
    out["a9_3b_LO_set"] = {"n": len(lo_ids), "tau_a9": a9tau, "a9_effect": eff(a9f["base"], a9f["black"], lo_ids)}
    val = table(rows, "val700")["base"]
    ms = sorted(margin(l) for l in val.values())
    tau = ms[int(0.15 * len(ms))]
    out["tau_m_model"] = tau
    f = table(rows, "fresh")
    ids2 = sorted(set(f["base"]) & set(f["black"]))
    out["FRESH"] = {"acc_true": round(acc(f, ids2), 4), "acc_black": round(acc(f, ids2, "black"), 4), "all": eff(f["base"], f["black"], ids2), "a9_acc_true": round(acc(a9f, ids2), 4)}
    out["on_a9_LO_set"] = eff(f["base"], f["black"], lo_ids)
    mb = {i: margin(f["black"][i]) for i in ids2}
    strat = {i: ("HI" if mb[i] >= tau else ("MID" if mb[i] >= 0.90 else "LO")) for i in ids2}
    S = {s: [i for i in ids2 if strat[i] == s] for s in ("HI", "MID", "LO")}
    out["own_strata"] = {"sizes": {s: len(v) for s, v in S.items()}, "by_stratum": {s: eff(f["base"], f["black"], v) for s, v in S.items() if v},
                         "HI_changed": sum(pred(f["base"][i]) != pred(f["black"][i]) for i in S["HI"])}
    te = table(rows, "test711")
    tid = sorted(set(te["base"]) & set(te["black"]))
    out["TEST711"] = {"acc_true": round(acc(te, tid), 4), "acc_black": round(acc(te, tid, "black"), 4), "pred_equals_shortest": round(sum(pred(te["base"][i]) == short_letter(i, "orig") for i in tid) / len(tid), 4)}
    a9_lo_true = out["a9_3b_LO_set"]["a9_effect"]["acc_true"]
    v = {"M2a_effect_ge5_on_a9_LO": sig(out["on_a9_LO_set"], 5.0),
         "M2b_LO_acc_gain_ge3": (out["on_a9_LO_set"]["acc_true"] - a9_lo_true) >= 0.03,
         "M2c_overall_acc_not_worse_than_1pt": (out["FRESH"]["acc_true"] - out["FRESH"]["a9_acc_true"]) >= -0.01}
    out["verdicts"] = v
json.dump(out, open(os.path.join(run_dir, "summary_a12.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
