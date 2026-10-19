#!/usr/bin/env python
"""A17: second, non-length shortcut: ANSWER POSITION. The correct option is placed at position A with probability p, otherwise at a uniformly random position among B-D;
the other options fill the remaining positions in random order. Option texts are not modified at all (no filler, no semantic content), so there is no formatting or length
change; p = 0.25 is the matched control (uniform random reordering with the same code path)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_rand import _rng  # noqa: E402


def pos_options(cid, opts, gold_idx, p):
    n = len(opts)
    r = _rng(cid, "pos")
    gp = 0 if r.random() < p else r.randint(1, n - 1)
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
    rr = random.Random(7)
    for p in (0.25, 0.55, 0.92):
        cnt, pos = 0, [0, 0, 0, 0]
        for t in range(8000):
            opts = [f"option {t} {k}" for k in range(4)]
            g = rr.randrange(4)
            new, gl = pos_options(f"q{t}", opts, g, p)
            assert sorted(new) == sorted(opts) and new["ABCD".index(gl)] == opts[g]
            pos["ABCD".index(gl)] += 1
        print(p, [round(x / 8000, 3) for x in pos])
    print("PROPERTY TESTS PASSED")
