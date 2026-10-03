// N=24 scored-path arm "fx2": QSB_FX3_PRESUB 2 — the fused X3 = R^2+PPP-2Q
// also drops the second fold's carry capture and the z3 carry (the lean
// products' Z2_SPEC_CUT class, ~2^-22; hit_filter_field_sc.cuh:52-56).
// Default is 1; value 2 is one step leaner. Exact host gate rejects the rare
// wrong tentative. Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_FX3_PRESUB 2
#include "subset.cu"
