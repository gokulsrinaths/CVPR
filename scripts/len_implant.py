#!/usr/bin/env python
"""A13: plant an option-length cue in a benchmark that has none (PMC-VQA), using the filler of len_rand.py on ALL options.
implant: the correct option gets final-length rank 1 (shortest) with probability P = 0.92, otherwise a uniformly random rank among 2..5; the wrong options take the remaining ranks in random order.
Final lengths are exactly s_max + 4j characters (j = 1..5 for five options; here four options: j = 1..4), every option carries filler (>= 4 characters)."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_rand import _rng, filler  # noqa: E402

P_CORRECT_SHORTEST = 0.92


def implant_options(cid, opts, gold_idx, p=P_CORRECT_SHORTEST):
    n = len(opts)
    lens = [len(o) for o in opts]
    smax = max(lens)
    r = _rng(cid, "implant")
    gold_rank = 1 if r.random() < p else r.randint(2, n)
    others = [j for j in range(1, n + 1) if j != gold_rank]
    r.shuffle(others)
    rank, k = {}, 0
    for i in range(n):
        if i == gold_idx:
            rank[i] = gold_rank
        else:
            rank[i] = others[k]
            k += 1
    return [opts[i] + filler(smax + 4 * rank[i] - lens[i], _rng(cid, f"ifill{i}")) for i in range(n)], rank


if __name__ == "__main__":
    import random
    rr = random.Random(5)
    cnt, tot = 0, 4000
    for t in range(tot):
        opts = [("y" * rr.randint(2, 70)) for _ in range(4)]
        g = rr.randrange(4)
        o, rank = implant_options(f"c{t}", opts, g)
        L = [len(x) for x in o]
        assert sorted(L) == [max(len(x) for x in opts) + 4 * j for j in range(1, 5)]
        assert all(len(o[i]) >= len(opts[i]) + 4 for i in range(4))
        cnt += L[g] == min(L)
    print("correct is shortest in", cnt / tot, "(target 0.92)")
    assert 0.90 < cnt / tot < 0.94
    print("PROPERTY TESTS PASSED")
