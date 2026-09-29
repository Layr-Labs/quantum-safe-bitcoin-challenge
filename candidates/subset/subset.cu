#define QSB_REDRAW_CEFIKA_G_0930 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
/* Host producers off (ercumentyildirim's QSB_HP_SKIP): the GPU producers build every batch and the co-grinder
 * gets their CPU time. With the producers off the main thread keeps its full CPU set, so the worker count
 * would fall to ncpu - QSB_CPU_RESERVE = 30; the main thread sleeps in blocking waits and the workers are
 * SCHED_IDLE, so no CPU is reserved: 32 workers. */
#define QSB_HP_SKIP 1
#define QSB_CPU_RESERVE 0
#include "tests/gpu_epochs/tree.cu"
