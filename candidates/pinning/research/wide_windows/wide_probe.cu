// Research-only native resource probe; not a submitted grinder.
// Includes the exact corrected PR24 primitives and recode setup.
#define main qsb_reference_main
#include "../pr24_corrected/pinning.cu"
#undef main
#include "wide_geometry.cuh"
__device__ void wide_fixed(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,const uint64_t k[4],const uint8_t *table){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    uint32_t idx;uint64_t neg;uint64_t x0[4],y0[4],x1[4],y1[4];
    int32_t e=wide_step(M,sign,26);gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,wide_offset(0),idx,neg,x0,y0);
    e=wide_step(M,sign,26);gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,wide_offset(1),idx,neg,x1,y1);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ,x0,y0,x1,y1);
    uint64_t x[4],y[4];unsigned offset=wide_offset(2);
    #pragma unroll 1
    for(int c=2;c<WIDE_CHUNKS-1;++c){
        e=wide_step(M,sign,wide_bits(c));gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,offset,idx,neg,x,y);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,x,y,y0);Load256(y0,y);offset+=wide_entries(c);
    }
    e=sign*(int32_t)M[0];gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,offset,idx,neg,x,y);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,x,y,y0);
}
template<bool WIDE> __global__ __launch_bounds__(256,2)
void qsb_window_probe(const uint64_t *scalars,const uint8_t *table,uint64_t *points,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4];
    if(WIDE)wide_fixed(X,Y,ZZ,ZZZ,scalars+(size_t)i*4,table);
    else _FixedBaseSignedXYZZScalar(X,Y,ZZ,ZZZ,scalars+(size_t)i*4,table);
    #pragma unroll
    for(int j=0;j<4;++j){points[(size_t)i*16+j]=X[j];points[(size_t)i*16+4+j]=Y[j];points[(size_t)i*16+8+j]=ZZ[j];points[(size_t)i*16+12+j]=ZZZ[j];}
}
// This program is build-only; the launch wrapper forces both instantiations.
void qsb_probe_launch(const uint64_t *s,const uint8_t *t,uint64_t *p,int n){
    qsb_window_probe<false><<<(n+255)/256,256>>>(s,t,p,n);
    qsb_window_probe<true><<<(n+255)/256,256>>>(s,t,p,n);
}
int main(){return 0;}
