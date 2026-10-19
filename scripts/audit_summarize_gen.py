#!/usr/bin/env python
"""Summary for H1' (amendment A3). python audit_summarize_gen.py RUN_DIR | selftest"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, wilson, mcnemar_exact, gold_from_train, shortest_letter


def summarize(run_dir):
    rows = read_jsonl(os.path.join(run_dir, "rows.jsonl"))
    T = gold_from_train()
    arm = {"base": {}, "black": {}}
    for r in rows:
        arm[r["arm"]][r["case_id"]] = r
    ids = sorted(set(arm["base"]) & set(arm["black"]))
    n = len(ids)
    short = {i: shortest_letter(T[i]) for i in ids}
    gold = {i: T[i]["answer"].strip() for i in ids}
    wrong_short = [i for i in ids if short[i] != gold[i]]
    pr = lambda a, i: arm[a][i]["parsed"]
    out = {"run_dir": run_dir, "n_cases": n}
    for a in ("base", "black"):
        k = sum(pr(a, i) == gold[i] for i in ids)
        out[a] = {"correct": k, "acc": round(k / n, 4), "wilson95": wilson(k, n),
                  "unparsed_rate": round(sum(pr(a, i) is None for i in ids) / n, 4),
                  "share_pred_equals_shortest": round(sum(pr(a, i) == short[i] for i in ids) / n, 4),
                  "acc_on_shortest_wrong_63": round(sum(pr(a, i) == gold[i] for i in wrong_short) / max(1, len(wrong_short)), 4),
                  "raw_examples": [arm[a][i]["raw"] for i in ids[:5]]}
    b = sum(pr("base", i) == gold[i] and pr("black", i) != gold[i] for i in ids)
    c = sum(pr("base", i) != gold[i] and pr("black", i) == gold[i] for i in ids)
    loss = round(100 * (out["base"]["acc"] - out["black"]["acc"]), 2)
    out["loss_points"], out["mcnemar_true_only"], out["mcnemar_black_only"], out["mcnemar_p"] = loss, b, c, mcnemar_exact(b, c)
    if out["base"]["acc"] < 0.60 or max(out["base"]["unparsed_rate"], out["black"]["unparsed_rate"]) > 0.10:
        out["H1prime"] = "NOT EVALUABLE (acc_true < 0.60 or unparsed > 10%)"
    else:
        out["H1prime"] = "SUPPORTED" if loss <= 5.0 else "FALSIFIED"
    return out


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        from audit_generate import parse  # noqa
        sys.exit(0)
    res = summarize(sys.argv[1])
    json.dump(res, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
