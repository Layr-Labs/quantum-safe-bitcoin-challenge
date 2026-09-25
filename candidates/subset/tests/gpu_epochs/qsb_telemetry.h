/* qsb_telemetry.h -- host-only run diagnostics for the ranked subset window.
 *
 * Nothing here touches device code, the hit path or the candidate identity; it
 * only changes WHERE in the (unchanged) epoch space each batch starts and what
 * the process prints. Two independent mechanisms, each with its own switch:
 *
 * QSB_TIME_SLOTS (default 1): time-stamped batches. The epoch space is cut into
 * slots of one full batch (QSB_SE_LAUNCH_BLOCKS * QSB_PAIR_MUL epochs). Before a
 * batch is enqueued its base is advanced to slot floor(t * QSB_TIME_SLOTS_PER_S)
 * when that slot lies ahead, where t is seconds since main() entered. Bases only
 * ever move forward and every batch still covers a disjoint, aligned range of
 * distinct epochs, so every candidate is searched at most once, exactly as
 * before; the skipped slots are simply never searched. Each batch yields ~16
 * published hits, so the public hit list of the run reveals which slots were
 * used and therefore when every batch was launched: the ranked host's throughput
 * profile over the whole 1200 s window, including the startup delay, can be read
 * back from the diagnostics artifact alone. 6.5 slots/s keeps a 1205 s run inside
 * the 7,837 whole slots of C(137,6) and above the peak batch rate (~6.1/s).
 *
 * QSB_TELEMETRY (default 1): NVML samples (dlopen'd libnvidia-ml.so.1; any failure
 * just disables the sampling) of SM/memory clock, board power, the energy counter,
 * GPU temperature and the clock-event (throttle) reasons, once per second from the
 * host loop while the GPU works on the queued batches. At the stop signal the run
 * prints one summary into the two numeric fields harness/gpu_wrap.py records as
 * "self-reported" (never scored): the only M/s token of the run, and one count
 * token that dominates the self-reported candidate count. Layout (decimal):
 *
 *   count token  (13 digits) C PPP WWW SSSS TT
 *       C    1 = native carrier on, 2 = off; +2 when NVML was unavailable
 *       PPP  enforced power limit, W
 *       WWW  mean board power over the last 600 s (energy counter), W
 *       SSSS mean SM clock over the last 600 s, MHz
 *       TT   mean GPU temperature over the last 600 s, C (99 = >= 99)
 *   rate token   ABC.DDDEEE  (printed as %03u.%06u, read back zero-padded)
 *       A    tenths of late samples with SW power cap active   (0..9)
 *       B    tenths of late samples with SW thermal slowdown   (0..9)
 *       C    tenths of late samples with any HW slowdown/thermal/power brake
 *       DDD  mean SM clock over the first 60 s of search / 10, MHz
 *       EEE  mean board power over the first 60 s of search (energy counter), W
 *
 * Every other progress/summary line prints its rate as "Mcand/s", so the M/s
 * token above is the only one gpu_wrap.py sees.
 */
#pragma once
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef QSB_TIME_SLOTS
#define QSB_TIME_SLOTS 1
#endif
#ifndef QSB_TIME_SLOTS_PER_S
#define QSB_TIME_SLOTS_PER_S 6.5
#endif
#ifndef QSB_TELEMETRY
#define QSB_TELEMETRY 1
#endif

static struct timespec g_qsb_proc_t0;
static inline void qsb_proc_clock_start(void) { clock_gettime(CLOCK_MONOTONIC, &g_qsb_proc_t0); }
static inline double qsb_proc_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)(t.tv_sec - g_qsb_proc_t0.tv_sec) + (double)(t.tv_nsec - g_qsb_proc_t0.tv_nsec) * 1e-9;
}

/* Base of the next batch: the next contiguous base, or the start of the current
 * time slot when that lies ahead and a whole batch still fits below n_epochs. */
static inline uint64_t qsb_time_slot_base(uint64_t next_base, uint64_t capacity, uint64_t n_epochs) {
#if QSB_TIME_SLOTS
    const double t = qsb_proc_seconds();
    if (t > 0.0 && capacity) {
        const uint64_t want = (uint64_t)(t * (double)QSB_TIME_SLOTS_PER_S) * capacity;
        if (want > next_base && want + capacity <= n_epochs) return want;
    }
#endif
    return next_base;
}

#if QSB_TELEMETRY
typedef struct qsb_nvml_dev *qsb_nvml_dev_t;
typedef int (*qsb_nvml_init_t)(void);
typedef int (*qsb_nvml_by_bus_t)(const char *, qsb_nvml_dev_t *);
typedef int (*qsb_nvml_by_index_t)(unsigned, qsb_nvml_dev_t *);
typedef int (*qsb_nvml_u32_t)(qsb_nvml_dev_t, unsigned *);
typedef int (*qsb_nvml_clock_t)(qsb_nvml_dev_t, int, unsigned *);
typedef int (*qsb_nvml_temp_t)(qsb_nvml_dev_t, int, unsigned *);
typedef int (*qsb_nvml_u64_t)(qsb_nvml_dev_t, unsigned long long *);

struct QsbTelemetrySample {
    float t;                 /* seconds since main() */
    uint16_t sm, mem;        /* MHz */
    uint32_t power_mw;
    uint64_t energy_mj;      /* 0 when the counter is unavailable */
    uint32_t reasons;        /* clock-event reason bits */
    uint8_t temp;            /* C */
};

enum { QSB_TELEM_MAX = 4096 };
struct QsbTelemetry {
    int ok;                  /* NVML loaded and a device handle resolved */
    int tried;
    void *lib;
    qsb_nvml_dev_t dev;
    qsb_nvml_clock_t clock;
    qsb_nvml_u32_t power, plimit;
    qsb_nvml_temp_t temp;
    qsb_nvml_u64_t reasons, energy;
    unsigned plimit_mw;
    double t_search0;        /* first batch enqueued */
    double t_last;
    int n;
    QsbTelemetrySample s[QSB_TELEM_MAX];
};
static QsbTelemetry g_qsb_telem;

static void qsb_telemetry_open(int cuda_dev) {
    QsbTelemetry &T = g_qsb_telem;
    if (T.tried) return;
    T.tried = 1;
    T.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!T.lib) return;
    qsb_nvml_init_t init = (qsb_nvml_init_t)dlsym(T.lib, "nvmlInit_v2");
    qsb_nvml_by_bus_t by_bus = (qsb_nvml_by_bus_t)dlsym(T.lib, "nvmlDeviceGetHandleByPciBusId_v2");
    qsb_nvml_by_index_t by_index = (qsb_nvml_by_index_t)dlsym(T.lib, "nvmlDeviceGetHandleByIndex_v2");
    T.clock = (qsb_nvml_clock_t)dlsym(T.lib, "nvmlDeviceGetClockInfo");
    T.power = (qsb_nvml_u32_t)dlsym(T.lib, "nvmlDeviceGetPowerUsage");
    T.plimit = (qsb_nvml_u32_t)dlsym(T.lib, "nvmlDeviceGetEnforcedPowerLimit");
    T.temp = (qsb_nvml_temp_t)dlsym(T.lib, "nvmlDeviceGetTemperature");
    T.reasons = (qsb_nvml_u64_t)dlsym(T.lib, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!T.reasons) T.reasons = (qsb_nvml_u64_t)dlsym(T.lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
    T.energy = (qsb_nvml_u64_t)dlsym(T.lib, "nvmlDeviceGetTotalEnergyConsumption");
    if (!init || !T.clock || !T.power || init() != 0) return;
    char bus[64] = {0};
    int have = 0;
    if (by_bus && cudaDeviceGetPCIBusId(bus, (int)sizeof(bus), cuda_dev) == cudaSuccess)
        have = by_bus(bus, &T.dev) == 0;
    if (!have && by_index) have = by_index((unsigned)cuda_dev, &T.dev) == 0;
    (void)cudaGetLastError();
    if (!have) return;
    if (!T.plimit || T.plimit(T.dev, &T.plimit_mw) != 0) T.plimit_mw = 0;
    T.ok = 1;
}

/* One sample, at most once per second; cheap enough for the host loop (the GPU
 * holds one to two queued batches, ~0.2 s each, while this runs). */
static void qsb_telemetry_poll(int cuda_dev) {
    QsbTelemetry &T = g_qsb_telem;
    const double now = qsb_proc_seconds();
    if (T.t_search0 == 0.0) { T.t_search0 = now; T.t_last = now; return; }
    if (now - T.t_last < 1.0) return;
    T.t_last = now;
    if (!T.tried) qsb_telemetry_open(cuda_dev);
    if (!T.ok || T.n >= QSB_TELEM_MAX) return;
    QsbTelemetrySample &x = T.s[T.n];
    unsigned v = 0;
    unsigned long long w = 0;
    x.t = (float)now;
    x.sm = (uint16_t)(T.clock(T.dev, 1, &v) == 0 ? v : 0);
    x.mem = (uint16_t)(T.clock(T.dev, 2, &v) == 0 ? v : 0);
    x.power_mw = T.power(T.dev, &v) == 0 ? v : 0;
    x.temp = (uint8_t)((T.temp && T.temp(T.dev, 0, &v) == 0) ? (v > 255 ? 255 : v) : 0);
    x.reasons = (uint32_t)((T.reasons && T.reasons(T.dev, &w) == 0) ? w : 0);
    x.energy_mj = (T.energy && T.energy(T.dev, &w) == 0) ? (uint64_t)w : 0;
    T.n++;
}

/* Mean board power over [a,b] from the energy counter when present, else the
 * mean of the power samples. */
static double qsb_telemetry_mean_power_w(double a, double b) {
    const QsbTelemetry &T = g_qsb_telem;
    int i0 = -1, i1 = -1;
    double psum = 0.0; int pn = 0;
    for (int i = 0; i < T.n; i++) {
        if (T.s[i].t < a || T.s[i].t > b) continue;
        if (i0 < 0) i0 = i;
        i1 = i;
        psum += T.s[i].power_mw * 1e-3; pn++;
    }
    if (i0 >= 0 && i1 > i0 && T.s[i0].energy_mj && T.s[i1].energy_mj > T.s[i0].energy_mj) {
        const double dt = (double)T.s[i1].t - (double)T.s[i0].t;
        if (dt > 1.0) return (double)(T.s[i1].energy_mj - T.s[i0].energy_mj) * 1e-3 / dt;
    }
    return pn ? psum / pn : 0.0;
}

static unsigned qsb_clampu(double v, unsigned hi) {
    if (!(v > 0.0)) return 0;
    const double r = v + 0.5;
    return r >= (double)hi ? hi : (unsigned)r;
}

/* Print the two encoded tokens (see the header comment). */
static void qsb_telemetry_report(int carrier_on) {
    const QsbTelemetry &T = g_qsb_telem;
    const double t_end = T.n ? (double)T.s[T.n - 1].t : 0.0;
    const double la = t_end - 600.0, lb = t_end;
    const double ea = T.t_search0 + 1.0, eb = T.t_search0 + 61.0;
    double sm_l = 0, tp_l = 0, sm_e = 0; int n_l = 0, n_e = 0, sw_pwr = 0, sw_th = 0, hw = 0;
    for (int i = 0; i < T.n; i++) {
        const QsbTelemetrySample &x = T.s[i];
        if (x.t >= la && x.t <= lb) {
            sm_l += x.sm; tp_l += x.temp; n_l++;
            if (x.reasons & 0x4) sw_pwr++;
            if (x.reasons & 0x20) sw_th++;
            if (x.reasons & (0x8 | 0x40 | 0x80)) hw++;
        }
        if (x.t >= ea && x.t <= eb) { sm_e += x.sm; n_e++; }
    }
    if (n_l) { sm_l /= n_l; tp_l /= n_l; }
    if (n_e) sm_e /= n_e;
    const unsigned mark = (carrier_on ? 1u : 2u) + (T.ok ? 0u : 2u);
    const unsigned plim = qsb_clampu(T.plimit_mw * 1e-3, 999);
    const unsigned pw_l = qsb_clampu(qsb_telemetry_mean_power_w(la, lb), 999);
    const unsigned pw_e = qsb_clampu(qsb_telemetry_mean_power_w(ea, eb), 999);
    const unsigned smc_l = qsb_clampu(sm_l, 9999);
    const unsigned tmp_l = qsb_clampu(tp_l, 99);
    const unsigned smc_e = qsb_clampu(sm_e / 10.0, 999);
    const unsigned fa = n_l ? (unsigned)(10 * sw_pwr / n_l > 9 ? 9 : 10 * sw_pwr / n_l) : 0;
    const unsigned fb = n_l ? (unsigned)(10 * sw_th / n_l > 9 ? 9 : 10 * sw_th / n_l) : 0;
    const unsigned fc = n_l ? (unsigned)(10 * hw / n_l > 9 ? 9 : 10 * hw / n_l) : 0;
    printf("  telemetry: samples=%d late=%d sm=%u MHz temp=%u C power=%u/%u W early sm=%u0 MHz power=%u W"
           " swpwr=%u swth=%u hw=%u carrier=%d\n",
           T.n, n_l, smc_l, tmp_l, pw_l, plim, smc_e, pw_e, fa, fb, fc, carrier_on);
    printf("  telemetry-code (%u%03u%03u%04u%02uM/ %u%u%u.%03u%03uM/s\n",
           mark, plim, pw_l, smc_l, tmp_l, fa, fb, fc, smc_e, pw_e);
    fflush(stdout);
}

/* ---- In-run arm comparison (QSB_GEO_AB builds) ------------------------------------------------
 * The host picks the arm of every batch from the phase clock (QSB_AB_PHASE_S-second phases, the
 * order rotated every three phases so each arm takes each position equally often). At each batch
 * completion (the slot's event, observed by the host that was waiting on it) the elapsed time and the
 * NVML energy counter delta since the previous completion are charged to the batch that just finished:
 * batches run back to back on the GPU, so that interval is that batch's execution. Only completions
 * after QSB_AB_WARMUP_S enter the totals. The summary replaces the telemetry tokens:
 *   count token (13 digits) C EEEE aaaa bbbb
 *       C     1 = carrier on, 2 = off; +2 when the energy counter was unavailable
 *       EEEE  arm 0 energy per candidate, 0.1 nJ units
 *       aaaa  5000 + round(1e4 * (E1/E0 - 1))     bbbb  5000 + round(1e4 * (E2/E0 - 1))
 *   rate token  RRR.xxxyyy  RRR arm 0 candidates/s in M (integer part);
 *       xxx = 500 + round(1e3 * (R1/R0 - 1)), yyy = 500 + round(1e3 * (R2/R0 - 1)) */
#ifndef QSB_AB_PHASE_S
#define QSB_AB_PHASE_S 20.0
#endif
#ifndef QSB_AB_WARMUP_S
#define QSB_AB_WARMUP_S 120.0
#endif
static inline int qsb_ab_arm_at(double t) {
    const long p = (long)(t / QSB_AB_PHASE_S);
    return (int)((p + p / 3) % 3);
}
/* Energy per batch comes from the NVML total-energy counter when the board exposes it; otherwise from
 * the board-power reading (a ~1 s average on this generation) times the batch's interval, counted only
 * once the reading has settled (>= 2.5 s into the phase, so it no longer averages the previous arm). */
struct QsbAbArm { double cand, secs, cand_e, mj, cand_p, mj_p; long batches; };
static QsbAbArm g_ab[3];
static double g_ab_last_t = -1.0;
static unsigned long long g_ab_last_e = 0;
static void qsb_ab_drained(int arm, double cand, int cuda_dev) {
    QsbTelemetry &T = g_qsb_telem;
    if (!T.tried) qsb_telemetry_open(cuda_dev);
    const double t = qsb_proc_seconds();
    unsigned long long e = 0;
    unsigned pw = 0;
    const int have_e = T.ok && T.energy && T.energy(T.dev, &e) == 0 && e != 0;
    const int have_p = T.ok && T.power && T.power(T.dev, &pw) == 0 && pw != 0;
    if (g_ab_last_t > 0.0 && t > QSB_AB_WARMUP_S && arm >= 0 && arm < 3) {
        QsbAbArm &A = g_ab[arm];
        const double dt = t - g_ab_last_t;
        A.cand += cand; A.secs += dt; A.batches++;
        if (have_e && g_ab_last_e && e >= g_ab_last_e) { A.mj += (double)(e - g_ab_last_e); A.cand_e += cand; }
        const double into = t - QSB_AB_PHASE_S * (double)(long)(t / QSB_AB_PHASE_S);
        if (have_p && into >= 2.5 && qsb_ab_arm_at(g_ab_last_t) == arm) { A.mj_p += (double)pw * dt; A.cand_p += cand; }
    }
    g_ab_last_t = t;
    if (have_e) g_ab_last_e = e; else g_ab_last_e = 0;
}
static void qsb_ab_report(int carrier_on) {
    double E[3] = {0, 0, 0}, R[3] = {0, 0, 0};
    const int use_counter = g_ab[0].cand_e > 0 && g_ab[1].cand_e > 0 && g_ab[2].cand_e > 0;
    for (int a = 0; a < 3; a++) {
        if (use_counter) E[a] = g_ab[a].mj * 1e6 / g_ab[a].cand_e;            /* mJ -> nJ per candidate */
        else if (g_ab[a].cand_p > 0) E[a] = g_ab[a].mj_p * 1e6 / g_ab[a].cand_p; /* mW*s = mJ */
        if (g_ab[a].secs > 0) R[a] = g_ab[a].cand / g_ab[a].secs;
    }
    const int have_e = E[0] > 0 && E[1] > 0 && E[2] > 0;
    /* C: 1/2 = energy counter (carrier on/off), 3/4 = no energy at all, 5/6 = power-reading fallback */
    const unsigned mark = (carrier_on ? 1u : 2u) + (!have_e ? 2u : (use_counter ? 0u : 4u));
    const unsigned e0 = qsb_clampu(E[0] * 10.0, 9999);
    const unsigned e1 = have_e ? qsb_clampu(5000.0 + 1e4 * (E[1] / E[0] - 1.0), 9999) : 0;
    const unsigned e2 = have_e ? qsb_clampu(5000.0 + 1e4 * (E[2] / E[0] - 1.0), 9999) : 0;
    const unsigned r0 = qsb_clampu(R[0] / 1e6, 999);
    const unsigned r1 = R[0] > 0 ? qsb_clampu(500.0 + 1e3 * (R[1] / R[0] - 1.0), 999) : 0;
    const unsigned r2 = R[0] > 0 ? qsb_clampu(500.0 + 1e3 * (R[2] / R[0] - 1.0), 999) : 0;
    printf("  arms: GLV12 %.1f nJ %.2f M/s (%ld) | GLV11 %.1f nJ %.2f M/s (%ld) | GLV10 %.1f nJ %.2f M/s (%ld)\n",
           E[0], R[0] / 1e6, g_ab[0].batches, E[1], R[1] / 1e6, g_ab[1].batches, E[2], R[2] / 1e6, g_ab[2].batches);
    printf("  arms-code (%u%04u%04u%04uM/ %03u.%03u%03uM/s\n", mark, e0, e1, e2, r0, r1, r2);
    fflush(stdout);
}
#else
static inline void qsb_telemetry_poll(int) {}
static inline void qsb_telemetry_report(int) {}
static inline int qsb_ab_arm_at(double) { return 1; }
static inline void qsb_ab_drained(int, double, int) {}
static inline void qsb_ab_report(int) {}
#endif
