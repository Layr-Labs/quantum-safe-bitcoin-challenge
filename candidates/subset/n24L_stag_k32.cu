// N=24 scored-path arm "k32": QSB_K32_SUBCUT 7 -> 3 — keep bits 1 (D = U2-X1
// sub3) and 2 (Q = X3-V sub14) of the seed's subtraction cuts but drop bit 4
// (the seed's three subtractions via qsb_filter_sub_cut). Bit 4's class is a
// dropped borrow into the high half (~2^-23/site). Arm prices whether bit 4's
// cut is net-positive or whether the seed's exact _ModSub256 is cheaper than
// the wrong-class risk load. Never measured on this tree.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_K32_SUBCUT 3
#include "subset.cu"
