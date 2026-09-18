"""Check tree arithmetic and shared-memory happens-before edges on the CPU.

This is a dependency model, not GPU execution or a performance measurement.
Every lane has a vector clock. A barrier joins clocks at its real scope.
Reads must observe the exact writer through program order or a barrier.
Negative controls deliberately weaken a cross-warp boundary.
"""
import random
from pathlib import Path

P = 2**256 - 2**32 - 977


class Memory:
    def __init__(self, n):
        self.n = n
        self.clock = [[0]*n for _ in range(n)]
        self.cells = {}

    def write(self, array, index, tid, value):
        self.clock[tid][tid] += 1
        self.cells[array, index] = (value, tid, self.clock[tid][tid])

    def read(self, array, index, tid):
        value, writer, stamp = self.cells[array, index]
        assert self.clock[tid][writer] >= stamp, (
            'unsynchronized read', array, index, tid, writer)
        return value

    def barrier(self, scope):
        for base in range(0, self.n, scope):
            lanes = range(base, min(base+scope, self.n))
            merged = [max(self.clock[t][j] for t in lanes)
                      for j in range(self.n)]
            for t in lanes:
                self.clock[t] = merged.copy()


def audit(values, u, weaken=None):
    n = len(values)
    m = Memory(n)
    for t, v in enumerate(values):
        m.write('p', t, t, v)
    m.barrier(n)
    h = [u[t]*m.read('p', t^(n//2), t) % P for t in range(n)]
    offset, count = 0, n
    checkpoint = {}
    while count > 1:
        half = count//2
        for t in range(half):
            v = m.read('p', offset+t, t)*m.read('p', offset+half+t, t) % P
            node = offset+count+t
            m.write('p', node, t, v)
            if node < 2*n-2:
                checkpoint[node-n] = v
        offset += count
        if count > 2:
            scope = 32 if count <= 64 else n
            if weaken == 'product' and count == 128:
                scope = 32
            m.barrier(scope)
        count //= 2
    root = m.read('p', 2*n-2, 0)
    assert root == __import__('functools').reduce(lambda a,b:a*b%P, values, 1)

    # A separate ordered kernel launch reloads the immutable checkpoint.
    m = Memory(n)
    for t in range(n-2):
        m.write('p', t, t, checkpoint[t])
    m.write('i', n-2, 0, pow(root, P-2, P))
    m.barrier(n)
    offset, count = 2*n-4, 2
    while count < n:
        half = count//2
        for t in range(count):
            parent = m.read('i', offset+count-n+(t&(half-1)), t)
            sibling = m.read('p', offset+(t^half)-n, t)
            m.write('i', offset-n+t, t, parent*sibling % P)
        offset -= 2*count
        scope = 32 if count < 32 else n
        if weaken == 'inverse' and count == 32:
            scope = 32
        if weaken == 'broadcast' and count == n//2:
            scope = 32
        m.barrier(scope)
        count *= 2
    got = [h[t]*m.read('i', t&(n//2-1), t) % P for t in range(n)]
    want = [u[t]*pow(values[t], P-2, P)%P for t in range(n)]
    assert got == want


def main():
    source = Path(__file__).with_name('LeafRecovery.cuh').read_text()
    assert 'if(count<=64)__syncwarp();' in source
    assert 'if(count<32)__syncwarp();' in source
    rng = random.Random(0x53434F5045)
    cases = 0
    for n in (128,256):
        boundaries = (0,1,31,32,33,63,64,65,n-1,n)
        for active in boundaries:
            raw = [rng.choice((0,1,2,P-2,P-1)) for _ in range(n)]
            u = [rng.randrange(1,P) if t<active and raw[t] else 0 for t in range(n)]
            values = [raw[t] if u[t] else 1 for t in range(n)]
            audit(values,u)
            cases += 1
        for _ in range(20):
            raw = [rng.randrange(1,P) for _ in range(n)]
            u = [rng.randrange(P) if rng.randrange(8) else 0 for _ in range(n)]
            audit([v if u[t] else 1 for t,v in enumerate(raw)],u)
            cases += 1
        for bad in ('product','inverse','broadcast'):
            try:
                audit(list(range(1,n+1)),[1]*n,weaken=bad)
            except AssertionError as e:
                assert e.args[0][0] == 'unsynchronized read', e
            else:
                raise AssertionError('negative control did not fail: '+bad)
    print(f'PASS: {cases} arithmetic/dependency cases; 6 unsafe barrier controls detected')


if __name__ == '__main__':
    main()
