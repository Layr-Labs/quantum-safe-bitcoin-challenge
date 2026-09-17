#!/usr/bin/env python3
"""Compare two subset hit files launch by launch.

usage: compare_hits.py <reference digest_hit file> <candidate digest_hit file> <epochs per launch>

Hits are keyed by (indices, recid). The six early omissions of a short-epoch
hit give its epoch's lexicographic rank in C(137,6); rank // epochs_per_launch is
its launch. Launches are compared up to the last launch with hits in the shorter
run, so the two runs may stop at different times. Exit status 1 on any mismatch."""
import re, sys
from math import comb

def hits(path):
    out, cur = [], None
    for line in open(path).read().split('\n'):
        m = re.match(r'indices=([0-9,]+)$', line)
        if m:
            cur = m.group(1); continue
        m = re.match(r'recid=(\d)', line)
        if m and cur is not None:
            out.append((cur, int(m.group(1)))); cur = None
    return out

def lex_rank(c, n):
    k, r, prev = len(c), 0, -1
    for i, x in enumerate(c):
        for y in range(prev + 1, x):
            r += comb(n - y - 1, k - i - 1)
        prev = x
    return r

def by_launch(hs, per):
    d = {}
    for idx, rid in hs:
        v = [int(x) for x in idx.split(',')]
        d.setdefault(lex_rank(v[:6], 137) // per, set()).add((idx, rid))
    return d

def main():
    ref, cand, per = hits(sys.argv[1]), hits(sys.argv[2]), int(sys.argv[3])
    assert lex_rank([0, 1, 2, 3, 4, 5], 137) == 0 and lex_rank([0, 1, 2, 3, 4, 6], 137) == 1
    if len(set(ref)) != len(ref) or len(set(cand)) != len(cand):
        print('FAIL: duplicate hits'); sys.exit(1)
    a, b = by_launch(ref, per), by_launch(cand, per)
    last = min(max(a, default=-1), max(b, default=-1))
    bad = [L for L in range(last + 1) if a.get(L, set()) != b.get(L, set())]
    n = sum(len(a.get(L, ())) for L in range(last + 1))
    print(f'launches compared: {last + 1}, hits compared: {n}, mismatching launches: {bad[:10]}')
    if bad or last < 0:
        print('FAIL'); sys.exit(1)
    print('PASS')

if __name__ == '__main__':
    main()
