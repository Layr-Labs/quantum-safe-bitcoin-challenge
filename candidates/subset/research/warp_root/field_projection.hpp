// Exact macros extracted from corrected3ac GPUMath; scalar PTX ops projected in host_support.hpp.
#define MM64 0xD838091DD2253531ULL
#define MSK62 0x3FFFFFFFFFFFFFFF
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}
#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}
