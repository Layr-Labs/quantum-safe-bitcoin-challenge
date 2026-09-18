/* Stein path (ZLAB_BY=0) compiled from the same shipped header text. */
#include <cstdint>
#include "warp.h"
namespace zst {
uint32_t zi_x(uint32_t v,int src);
#include "../tests/gpu_epochs/zinv32.cuh"
uint32_t zi_x(uint32_t v,int src){ return qsb_warp_shfl(v,src); }
}
void st_inverse(uint64_t *R,int lane){ zst::zi_inverse_quad(R,lane); }
