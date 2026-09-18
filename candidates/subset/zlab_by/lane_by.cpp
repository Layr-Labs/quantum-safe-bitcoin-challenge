/* Bernstein-Yang path (ZLAB_BY=1) compiled from the shipped header text. */
#include <cstdint>
#include "warp.h"
namespace zby {
uint32_t zi_x(uint32_t v,int src);
#include "../tests/gpu_epochs/zinv32.cuh"
uint32_t zi_x(uint32_t v,int src){ return qsb_warp_shfl(v,src); }
}
void by_inverse(uint64_t *R,int lane){ zby::zi_inverse_quad(R,lane); }
int32_t by_divstep30(int32_t delta,uint32_t f,uint32_t g,
                     int32_t *a,int32_t *b,int32_t *c,int32_t *d){
    return zby::zi_divstep30_by(delta,f,g,a,b,c,d);
}
