B=1<<256
K=(1<<32)+977
P=B-K
mask32=(1<<32)-1
def sm(a, b):
    f = a * b
    f = (f & B - 1) + K * (f >> 256)
    r = f & B - 1
    h = f >> 256 & mask32
    return r & ~((1 << 96) - 1) | r + h * K & (1 << 96) - 1

def original(vals, m):
    N = len(vals)
    pr = dict(enumerate(vals))
    ex = {}
    off = 0
    count = N
    while count > 2:
        half = count // 2
        pr.update({off + count + t: m(pr[off + t], pr[off + half + t]) for t in range(half)})
        off += count
        count //= 2
    root = m(pr[2 * N - 4], pr[2 * N - 3])
    ex.update({N - 8 + t: m(pr[2 * N - 4 + (t & 1 ^ 1)], pr[2 * N - 8 + (t ^ 2)]) for t in range(4)})
    off = 2 * N - 16
    count = 8
    while count < N:
        half = count // 2
        ex.update({off - N + t: m(ex[off + count - N + (t & half - 1)], pr[off + (t ^ half)]) for t in range(count)})
        off -= 2 * count
        count *= 2
    return (root, [m(ex[t & N // 2 - 1], pr[t ^ N // 2]) for t in range(N)])
