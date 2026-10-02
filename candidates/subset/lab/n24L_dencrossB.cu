// B-only relocation: avoid A's extra shared-memory round trip while retiring WA early.
#define QSB_LOCAL_SM86 1
#define QSB_DEN_CROSS_PRE 2
#include "../subset.cu"
