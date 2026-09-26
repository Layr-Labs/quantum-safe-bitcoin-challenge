#pragma once
/* Include inside namespace qcpu, after Ctx and CpuGlvScalar.h.
 * Requires canonical fe/pt helpers, batch_add, pt_double, std::vector/thread/atomic.
 * The caller owns c.cfold[2 * CPU_GLV_JOINT_ENTRIES] and fallback/free on false.
 * This helper never changes table metadata or publishes a partially built table.
 */

static bool cpu_glv_joint_on_curve(const pt &a) {
    if (fe_ge_p(a.x.v) || fe_ge_p(a.y.v)) return false;
    const fe seven = {{7, 0, 0, 0}};
    fe rhs, lhs;
    fe_sqr(rhs, a.x); fe_mul(rhs, rhs, a.x); fe_add(rhs, rhs, seven);
    fe_sqr(lhs, a.y);
    return fe_eq(lhs, rhs);
}

/* batch_add leaves a colliding input unchanged and sets bad[k]. Repair a true
 * doubling; an inverse-point pair would produce infinity, which an affine table
 * cannot encode. Returning false invalidates the whole joint table. Other rows
 * are never used as input to a bad row and cannot be poisoned by its denominator.
 */
static bool cpu_glv_joint_batch_exact(pt *out, const pt *const *tp, int n,
                                      fe *d, fe *pre, uint8_t *inf, uint8_t *bad) {
    for (int k = 0; k < n; ++k) { inf[k] = 0; bad[k] = 0; }
    batch_add(out, tp, inf, bad, n, d, pre);
    for (int k = 0; k < n; ++k) {
        if (!bad[k]) continue;
        if (!tp[k] || !fe_eq(out[k].x, tp[k]->x) ||
            !fe_eq(out[k].y, tp[k]->y) || fe_is_zero(out[k].y)) return false;
        out[k] = pt_double(out[k]);
    }
    return true;
}

/* Build T(x,y)+C followed by T(x,y)-C, where T=x*A+y*phi(A).
 * Dense rows: 0..128 are (x=row,y=0). Remaining rows have y=1..128,
 * x=-128..128, with row=129+(y-1)*257+(x+128).
 * Row zero is assigned C/-C directly; no infinity sentinel enters affine math.
 */
static bool build_glv_joint(Ctx &c, const fe &ax, const fe &ay, int nth) {
    enum { H = 128, WIDTH = 2 * H + 1, ROWS = H + 1 + H * WIDTH, CH = 1024 };
    static_assert(CPU_GLV_LOW_HALF == H, "joint builder requires w8 low digits");
    static_assert(CPU_GLV_JOINT_ENTRIES == ROWS, "joint row count mismatch");
    if (!c.cfold) return false;
    const pt A = {ax, ay}, C = {c.cx, c.cy};
    if (!cpu_glv_joint_on_curve(A) || !cpu_glv_joint_on_curve(C)) return false;
    const fe zero = {{0, 0, 0, 0}};
    const fe beta = {{CPU_GLV_BETA_LE64[0], CPU_GLV_BETA_LE64[1],
                      CPU_GLV_BETA_LE64[2], CPU_GLV_BETA_LE64[3]}};
    try {
        /* axis[H+x] = x*A for nonzero x. The center is never read. */
        std::vector<pt> axis(WIDTH), phi(H + 1), anchor(2 * (H + 1));
        std::vector<fe> sd(2 * (H + 1)), sp(2 * (H + 1));
        std::vector<uint8_t> si(2 * (H + 1)), sb(2 * (H + 1));
        std::vector<const pt *> st(2 * (H + 1));
        axis[H + 1] = A;
        for (int have = 1; have < H; have *= 2) {
            for (int k = 1; k <= have; ++k) {
                axis[H + have + k] = axis[H + k];
                st[k - 1] = &axis[H + have];
            }
            if (!cpu_glv_joint_batch_exact(&axis[H + have + 1], st.data(), have,
                                           sd.data(), sp.data(), si.data(), sb.data())) return false;
        }
        for (int k = 1; k <= H; ++k) {
            axis[H - k] = axis[H + k];
            fe_sub(axis[H - k].y, zero, axis[H + k].y);
            phi[k] = axis[H + k];
            fe_mul(phi[k].x, phi[k].x, beta);
        }
        /* Sanity-check the shared beta constant; its lambda orientation is shared with CpuGlvScalar.h. */
        if (!cpu_glv_joint_on_curve(phi[1]) || fe_eq(phi[1].x, A.x)) return false;

        pt cm = C; fe_sub(cm.y, zero, C.y);
        for (int m = 0; m < 2; ++m) {
            for (int y = 0; y <= H; ++y) {
                const int e = m * (H + 1) + y;
                anchor[e] = m ? cm : C;
                st[e] = y ? &phi[y] : nullptr;
            }
        }
        if (!cpu_glv_joint_batch_exact(anchor.data(), st.data(), 2 * (H + 1),
                                       sd.data(), sp.data(), si.data(), sb.data())) return false;
        /* Every anchor is itself an x=0 final row, so an infinite anchor would have
         * already returned false. The two y=0 anchors are the exact row-zero outputs. */
        c.cfold[0] = C;
        c.cfold[ROWS] = cm;

        const size_t total = (size_t)2 * ROWS, tasks = (total + CH - 1) / CH;
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        auto work = [&]() { try {
            std::vector<fe> d(CH), pre(CH);
            std::vector<uint8_t> inf(CH), bad(CH);
            std::vector<const pt *> tp(CH);
            for (;;) {
                if (failed.load(std::memory_order_relaxed)) break;
                const size_t task = next.fetch_add(1, std::memory_order_relaxed);
                if (task >= tasks) break;
                const size_t begin = task * CH;
                const int count = (int)(total - begin < CH ? total - begin : CH);
                pt *out = c.cfold + begin;
                for (int k = 0; k < count; ++k) {
                    const size_t e = begin + (size_t)k;
                    const int m = (int)(e / ROWS), row = (int)(e % ROWS);
                    int x, y;
                    if (row <= H) { x = row; y = 0; }
                    else { const int u = row - (H + 1); x = u % WIDTH - H; y = u / WIDTH + 1; }
                    out[k] = anchor[m * (H + 1) + y];
                    tp[k] = x ? &axis[H + x] : nullptr;
                }
                if (!cpu_glv_joint_batch_exact(out, tp.data(), count,
                                               d.data(), pre.data(), inf.data(), bad.data())) {
                    failed.store(true, std::memory_order_relaxed); break;
                }
            }
        } catch (...) { failed.store(true, std::memory_order_relaxed); } };
        int workers = nth > 0 ? nth : 1;
        if ((size_t)workers > tasks) workers = (int)tasks;
        std::vector<std::thread> threads;
        try { threads.reserve((size_t)(workers - 1)); } catch (...) { workers = 1; }
        for (int t = 1; t < workers; ++t) {
            try { threads.emplace_back(work); } catch (...) { break; }
        }
        work();
        for (auto &t : threads) t.join();
        return !failed.load(std::memory_order_relaxed);
    } catch (...) { return false; }
}
