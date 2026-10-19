#!/usr/bin/env python
"""A17: dose-controlled planting of a nonvisual shortcut with EXACT realised rate (systematic assignment).
Within a question set, each question gets u = (hash-rank + 0.5) / N in [0, 1); the correct option takes the shortcut position/length iff u < d, so the realised rate is d to within 1/N and
the set of cued questions is nested across doses. Otherwise the correct option takes a uniformly random non-cued rank/position. Length shortcut: neutral filler on ALL options (A12/A13
filler), final lengths s_max + 4j; position shortcut: option texts unchanged, only their order."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_rand import _rng, filler  # noqa: E402


def assign_u(ids):
    order = sorted(ids, key=lambda c: hashlib.sha256(("A17u|" + c).encode()).hexdigest())
    n = len(order)
    return {c: (k + 0.5) / n for k, c in enumerate(order)}


def length_options(cid, opts, gold_idx, u, d):
    n = len(opts)
    lens = [len(o) for o in opts]
    smax = max(lens)
    r = _rng(cid, "dose")
    gold_rank = 1 if u < d else r.randint(2, n)
    others = [j for j in range(1, n + 1) if j != gold_rank]
    r.shuffle(others)
    rank, k = {}, 0
    for i in range(n):
        if i == gold_idx:
            rank[i] = gold_rank
        else:
            rank[i] = others[k]
            k += 1
    return [opts[i] + filler(smax + 4 * rank[i] - lens[i], _rng(cid, f"dfill{i}")) for i in range(n)]


def position_options(cid, opts, gold_idx, u, d):
    n = len(opts)
    r = _rng(cid, "dpos")
    gp = 0 if u < d else r.randint(1, n - 1)
    others = [i for i in range(n) if i != gold_idx]
    r.shuffle(others)
    new = [None] * n
    new[gp] = opts[gold_idx]
    rest = [j for j in range(n) if j != gp]
    for j, i in zip(rest, others):
        new[j] = opts[i]
    return new, "ABCDE"[gp]


if __name__ == "__main__":
    import random
    rr = random.Random(3)
    ids = [f"q{t}" for t in range(2000)]
    u = assign_u(ids)
    for d in (0.25, 0.40, 0.55, 0.70, 0.85, 0.92):
        cl = cp = 0
        for c in ids:
            opts = [("z" * rr.randint(2, 60)) for _ in range(4)]
            g = rr.randrange(4)
            o = length_options(c, opts, g, u[c], d)
            L = [len(x) for x in o]
            assert sorted(L) == [max(len(x) for x in opts) + 4 * j for j in range(1, 5)] and all(len(o[i]) >= len(opts[i]) + 4 for i in range(4))
            cl += (L[g] == min(L))
            new, gl = position_options(c, opts, g, u[c], d)
            assert sorted(new) == sorted(opts) and new["ABCD".index(gl)] == opts[g]
            cp += (gl == "A")
        print(d, "length rate", cl / 2000, "position rate", cp / 2000)
        assert abs(cl / 2000 - d) <= 0.0006 and abs(cp / 2000 - d) <= 0.0006
    print("PROPERTY TESTS PASSED")
