#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#ifndef QSB_SINGLE_EPOCH
#define QSB_SINGLE_EPOCH 1
#endif
#ifndef QSB_SINGLE_THREADS
#define QSB_SINGLE_THREADS 192
#endif
#ifndef QSB_SINGLE_BLOCKS
#define QSB_SINGLE_BLOCKS 3
#endif
#ifndef QSB_SINGLE_MAXREG
#define QSB_SINGLE_MAXREG 96
#endif
#ifndef QSB_SINGLE_INLINE
#define QSB_SINGLE_INLINE 2
#endif
#ifndef QSB_SINGLE_DIGITS
#define QSB_SINGLE_DIGITS 1
#endif
#ifndef QSB_SINGLE_POINT_PARK
#define QSB_SINGLE_POINT_PARK 2
#endif
#include "tests/gpu_epochs/tree.cu"
