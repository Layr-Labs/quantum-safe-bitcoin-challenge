// N=24 scored-path arm "rcb": QSB_R_CBANK_TAILS 1 — tails read the original R
// from the constant bank inside the __noinline__ callee instead of taking eight
// 64-bit ABI args; kernel_digest stops reloading R after the tree and keeps its
// 16 registers across both tails. Documented bit-identical. Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_R_CBANK_TAILS 1
#include "subset.cu"
