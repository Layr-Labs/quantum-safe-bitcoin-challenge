/* Host-only launch sizing. Kernels, native fingerprint and epoch order are
 * unchanged: both producers and consumers receive the same selected capacity.
 * Only known 128-window / two-slot shapes opt in at the call site. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef QSB_LAUNCH_BUDGET
#define QSB_LAUNCH_BUDGET 0
#endif
#ifndef QSB_LAUNCH_RESERVE_MIB
#define QSB_LAUNCH_RESERVE_MIB 1024
#endif
/* Test-only ceiling; zero leaves the compiled full capacity available. */
#ifndef QSB_LAUNCH_TEST_MAX_EPOCHS
#define QSB_LAUNCH_TEST_MAX_EPOCHS 0
#endif
static size_t qsb_launch_buffer_bytes(size_t epochs, size_t groups,
                                     size_t descriptor_bytes, size_t group_bytes,
                                     size_t first_row_bytes) {
    /* The call site restricts epochs to at most the compiled 2^21 capacity. */
    return 2 * (epochs * (descriptor_bytes + first_row_bytes + sizeof(uint32_t)) +
                groups * group_bytes);
}
template<class GroupCapacity>
static size_t qsb_select_launch_epochs(size_t compiled, size_t free_bytes,
                                      size_t reserve_bytes, size_t ceiling,
                                      size_t descriptor_bytes, size_t group_bytes,
                                      size_t first_row_bytes, GroupCapacity groups) {
    size_t e = compiled;
    if (ceiling && ceiling < e) e = ceiling;
    /* Host producer pieces require NPIECE*CHUNK=2^17. Both supported caps
     * are multiples, so no producer disabling or checker coverage sampling. */
    if (e != 1048576 && e != 2097152) return compiled;
    while (e >= 1048576) {
        const size_t need = qsb_launch_buffer_bytes(e, groups(e), descriptor_bytes,
                                                  group_bytes, first_row_bytes);
        if (free_bytes >= reserve_bytes && need <= free_bytes - reserve_bytes) return e;
        e /= 2;
    }
    return 0; /* insufficient even at the minimum; fail with a diagnostic */
}
