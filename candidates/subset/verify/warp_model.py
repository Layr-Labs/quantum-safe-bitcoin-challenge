#!/usr/bin/env python3
"""Lane/round visibility model for the checkpointed product trees with warp barriers.

A CTA has 256 lanes in warps of 32. Execution is modeled in rounds separated by
barriers. A shared cell written by lane L in round r is readable by lane M in
round r' > r only if some barrier ending a round in [r, r'-1] synchronizes L and
M: a full barrier (__syncthreads) always does, a warp barrier (__syncwarp) only
when L and M share a warp. A lane always sees its own earlier writes. Every read
must also hit a written cell. Values are real field elements, so the model also
checks the returned inverses."""
import random
P = 2**256 - 2**32 - 977
LANES, WARP = 256, 32

class Shared:
    def __init__(self):
        self.cells = {}          # key -> (value, lane, round)
        self.barriers = []       # kind of barrier ending round i: 'full' / 'warp'
        self.round = 0
        self.reads = 0
    def write(self, key, value, lane):
        self.cells[key] = (value, lane, self.round)
    def read(self, key, lane):
        assert key in self.cells, f'read of unwritten cell {key} by lane {lane} in round {self.round}'
        value, writer, r = self.cells[key]
        if writer != lane:
            ok = any(kind == 'full' or (kind == 'warp' and writer // WARP == lane // WARP)
                     for kind in self.barriers[r:self.round])
            assert ok, (f'lane {lane} round {self.round} reads {key} written by lane {writer} '
                        f'in round {r}; barriers since: {self.barriers[r:self.round]}')
        self.reads += 1
        return value
    def barrier(self, kind):
        self.barriers.append(kind)
        self.round += 1

def product_checkpoint(values, up_rule):
    """values: 256 leaves. Returns (checkpoint dict node->value, root). Mirrors
    qsb_block_product_checkpoint; up_rule(count) -> 'full'/'warp'/None."""
    sh = Shared()
    checkpoint = {}
    for tid in range(LANES):
        sh.write(('p', tid), values[tid], tid)
    sh.barrier('full')
    offset, count = 0, 256
    root = None
    while count > 1:
        half = count >> 1
        for tid in range(LANES):
            if tid < half:
                a = sh.read(('p', offset + tid), tid)
                b = sh.read(('p', offset + half + tid), tid)
                node = offset + count + tid
                sh.write(('p', node), a * b % P, tid)
                if node < 510:
                    checkpoint[node - 256] = a * b % P
        offset += count
        kind = up_rule(count)
        if kind:
            sh.barrier(kind)
        count >>= 1
    root = sh.read(('p', 510), 0)          # lane 0, after the loop
    return checkpoint, root, sh

def inverse_checkpoint(values, checkpoint, root_inv, down_rule):
    """Mirrors qsb_block_inverse_checkpoint. Returns per-lane leaf inverses."""
    sh = Shared()
    for tid in range(LANES):
        sh.write(('p', tid), values[tid], tid)
        if tid < 254:
            sh.write(('p', 256 + tid), checkpoint[tid], tid)
        if tid == 0:
            sh.write(('i', 254), root_inv, tid)
    sh.barrier('full')
    offset, count = 508, 2
    while count < 256:
        half = count >> 1
        for tid in range(LANES):
            if tid < count:
                lp = tid & (half - 1)
                parent = sh.read(('i', offset + count - 256 + lp), tid)
                sibling = sh.read(('p', offset + (tid ^ half)), tid)
                sh.write(('i', offset - 256 + tid), parent * sibling % P, tid)
        offset -= count << 1
        sh.barrier(down_rule(count))
        count <<= 1
    out = []
    for tid in range(LANES):
        parent = sh.read(('i', tid & 127), tid)
        sibling = sh.read(('p', tid ^ 128), tid)
        out.append(parent * sibling % P)
    return out, sh

def block_inverse_tree(values, up_rule_w, down_rule_w):
    """Mirrors the promoted qsb_block_inverse_tree (d277241) for n=256."""
    sh = Shared(); n = LANES
    for tid in range(n):
        sh.write(('t', n + tid), values[tid], tid)
    sh.barrier('full')
    width = n >> 1
    while width > 0:
        for tid in range(n):
            if tid < width:
                node = width + tid
                a = sh.read(('t', 2 * node), tid); b = sh.read(('t', 2 * node + 1), tid)
                sh.write(('t', node), a * b % P, tid)
        sh.barrier(up_rule_w(width))
        width >>= 1
    root = sh.read(('t', 1), 0)
    sh.write(('t', 1), pow(root, -1, P), 0)
    sh.barrier('full')
    width = 1
    while width < n:
        for tid in range(n):
            if tid < width:
                node = width + tid
                parent = sh.read(('t', node), tid)
                left = sh.read(('t', 2 * node), tid); right = sh.read(('t', 2 * node + 1), tid)
                sh.write(('t', 2 * node), parent * right % P, tid)
                sh.write(('t', 2 * node + 1), parent * left % P, tid)
        sh.barrier(down_rule_w(width))
        width <<= 1
    return [sh.read(('t', n + tid), tid) for tid in range(n)], sh

# production rules
PIPE_UP = lambda count: 'full' if count > 64 else 'warp'
PIPE_DOWN = lambda count: 'full' if count >= 32 else 'warp'
REC_UP = lambda width: 'full' if width > 32 else 'warp'
REC_DOWN = lambda width: 'full' if (width << 1) > 32 else 'warp'
ALL_FULL_UP = lambda count: 'full' if count > 2 else None
ALL_FULL_DOWN = lambda count: 'full'

def check(up, down, trials, rng):
    for _ in range(trials):
        vals = [rng.randrange(1, P) for _ in range(LANES)]
        # inactive/identity lanes as in production
        for i in rng.sample(range(LANES), rng.randrange(0, 40)):
            vals[i] = 1
        cp, root, sh1 = product_checkpoint(vals, up)
        prod = 1
        for v in vals: prod = prod * v % P
        assert root == prod
        inv, sh2 = inverse_checkpoint(vals, cp, pow(root, -1, P), down)
        for v, w in zip(vals, inv):
            assert v * w % P == 1
    return sh1, sh2

def expect_failure(fn):
    try:
        fn()
    except AssertionError as e:
        return str(e)
    raise SystemExit('model failed to flag an invalid schedule')

def main():
    rng = random.Random(0x5EED)
    sh1, sh2 = check(PIPE_UP, PIPE_DOWN, 20, rng)
    kinds = lambda sh: (sh.barriers.count('full'), sh.barriers.count('warp'))
    print('pipeline prepare tree barriers (full, warp):', kinds(sh1), ' finish tree:', kinds(sh2))
    check(ALL_FULL_UP, ALL_FULL_DOWN, 5, rng)
    vals = [rng.randrange(1, P) for _ in range(LANES)]
    out, sh = block_inverse_tree(vals, REC_UP, REC_DOWN)
    assert all(v * w % P == 1 for v, w in zip(vals, out))
    print('promoted d277241 in-kernel tree under the same model: PASS', kinds(sh))
    # negative controls: one level too eager must be caught
    msg1 = expect_failure(lambda: check(lambda c: 'full' if c > 128 else 'warp', PIPE_DOWN, 1, rng))
    msg2 = expect_failure(lambda: check(PIPE_UP, lambda c: 'full' if c >= 64 else 'warp', 1, rng))
    msg3 = expect_failure(lambda: block_inverse_tree(vals, lambda w: 'full' if w > 64 else 'warp', REC_DOWN))
    print('negative controls flagged:\n  ', msg1[:110], '\n  ', msg2[:110], '\n  ', msg3[:110])
    print('PASS: warp-barrier checkpoint trees are visibility-safe and exact')

if __name__ == '__main__':
    main()
