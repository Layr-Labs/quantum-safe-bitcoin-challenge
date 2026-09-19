// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The grouped-prefix proof guarantees a finite accumulator before the final
// joint affine term. If their x coordinates agree, unequal y means infinity;
// equal y means doubling that known affine term. No projective inversion or
// reconstruction of the old accumulator is needed on this exceptional path.
// Output Y remains deferred at the final affine y, as the caller expects.
__device__ __forceinline__ void q10_final_exception(uint64_t*X,uint64_t*Y,uint64_t*U,uint64_t*V,
    const uint64_t*ax,const uint64_t*ay,const uint64_t*R){
    if(!q9_zero(R)){
        #pragma unroll
        for(int j=0;j<4;j++)X[j]=Y[j]=U[j]=V[j]=0;
        return;
    }
    uint64_t h[4],m[4],Q[4],T[4];
    _ModAdd256(h,(uint64_t*)ay,(uint64_t*)ay);
    qsb_field_mul(U,h,h);qsb_field_mul(V,U,h);
    qsb_field_mul(Q,(uint64_t*)ax,U);
    qsb_field_mul(m,(uint64_t*)ax,(uint64_t*)ax);
    _ModAdd256(T,m,m);_ModAdd256(m,T,m);
    qsb_field_mul(T,m,m);_ModSub256(X,T,Q);_ModSub256(X,X,Q);
    _ModSub256(T,Q,X);qsb_field_mul(Y,m,T);
}
