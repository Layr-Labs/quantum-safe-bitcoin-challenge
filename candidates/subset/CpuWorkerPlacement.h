#pragma once
/* Conservative affinity only; independent of host-producer state and arithmetic. */
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <ctype.h>
#ifdef CPU_COUNT
namespace qcpu_place {
/* Accept complete Linux CPU lists, with no silent truncation or malformed ranges. */
static bool parse_core(const char *text, int cpu, cpu_set_t *core) {
    CPU_ZERO(core);
    const char *p = text;
    while (*p && !isspace((unsigned char)*p)) {
        if (!isdigit((unsigned char)*p)) return false;
        char *e; errno = 0; long a = strtol(p, &e, 10), b = a;
        if (errno || a < 0 || a >= CPU_SETSIZE) return false;
        p = e;
        if (*p == '-') {
            if (!isdigit((unsigned char)*++p)) return false;
            errno = 0; b = strtol(p, &e, 10);
            if (errno || b < a || b >= CPU_SETSIZE) return false;
            p = e;
        }
        for (long x = a; x <= b; ++x) {
            if (CPU_ISSET(x, core)) return false;
            CPU_SET(x, core);
        }
        if (*p == ',') { ++p; if (!isdigit((unsigned char)*p)) return false; }
        else break;
    }
    while (isspace((unsigned char)*p)) ++p;
    return !*p && cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, core);
}
static bool read_core(int cpu, cpu_set_t *core) {
    char path[160], buf[8192];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE *f = fopen(path, "r"); if (!f) return false;
    const bool ok = fgets(buf, sizeof buf, f) != nullptr && fgetc(f) == EOF;
    fclose(f); return ok && parse_core(buf, cpu, core);
}
/* The fallback is exactly allowed-minus-main. Only a complete, symmetric
 * topology can narrow it further. This never re-adds main or increases a count. */
static bool conservative_mask(const cpu_set_t &allowed, const cpu_set_t &main,
                              cpu_set_t *work,
                              bool (*reader)(int, cpu_set_t *) = read_core) {
    CPU_ZERO(work);
    if (CPU_COUNT(&main) < 1 || CPU_COUNT(&main) >= CPU_COUNT(&allowed)) return false;
    for (int x = 0; x < CPU_SETSIZE; ++x) {
        if (CPU_ISSET(x, &main) && !CPU_ISSET(x, &allowed)) return false;
        if (CPU_ISSET(x, &allowed) && !CPU_ISSET(x, &main)) CPU_SET(x, work);
    }
    cpu_set_t candidate = *work;
    for (int x = 0; x < CPU_SETSIZE; ++x) if (CPU_ISSET(x, &main)) {
        cpu_set_t core;
        if (!reader(x, &core) || !CPU_ISSET(x, &core)) return true;
        for (int y = 0; y < CPU_SETSIZE; ++y) if (CPU_ISSET(y, &core)) {
            cpu_set_t reverse;
            if (!reader(y, &reverse) || !CPU_EQUAL(&core, &reverse)) return true;
            CPU_CLR(y, &candidate);
        }
    }
    if (CPU_COUNT(&candidate) > 0) *work = candidate;
    return true;
}
} // namespace qcpu_place
#endif
