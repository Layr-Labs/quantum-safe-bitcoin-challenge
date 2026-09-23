// GPL-3.0-only; constant-address form of VanitySearch's _DivStep62 head
// selection. See GPUMath.h for Jean Luc Pons's original copyright notice.
#pragma once
__device__ __forceinline__ void qsb_divstep_head(const uint64_t u[5], const uint64_t v[5],
                                               int32_t *pos, uint64_t *uh, uint64_t *vh) {
    const int bound=*pos;
    uint64_t hiu,hiv,lou,lov;
    int p;
    if(bound>=4 && (u[4]|v[4])) {
        p=4; hiu=u[4]; hiv=v[4]; lou=u[3]; lov=v[3];
    } else if(bound>=3 && (u[3]|v[3])) {
        p=3; hiu=u[3]; hiv=v[3]; lou=u[2]; lov=v[2];
    } else if(bound>=2 && (u[2]|v[2])) {
        p=2; hiu=u[2]; hiv=v[2]; lou=u[1]; lov=v[1];
    } else if(bound>=1 && (u[1]|v[1])) {
        p=1; hiu=u[1]; hiv=v[1]; lou=u[0]; lov=v[0];
    } else {
        p=0; hiu=u[0]; hiv=v[0]; lou=lov=0;
    }
    *pos=p;
    if(p==0) { *uh=hiu; *vh=hiv; return; }
    const unsigned shift=__clzll(hiu|hiv);
    if(shift==0) { *uh=hiu; *vh=hiv; return; }
    *uh=(hiu<<shift)|(lou>>(64-shift));
    *vh=(hiv<<shift)|(lov>>(64-shift));
}

__device__ __forceinline__ void hm39_divstep62(uint64_t u[5], uint64_t v[5],
                           int32_t *pos,
                           int64_t *uu, int64_t *uv,
                           int64_t *vu, int64_t *vv)
{


    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of
    // 'added P' can be easily calculated

    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;

    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    // Extract 64 MSB of u and v
    // u and v must be positive
    uint64_t uh, vh;
    int64_t w, x, y, z;
    bitCount = 62;

    qsb_divstep_head(u,v,pos,&uh,&vh);

    while (true) {

        // Use a sentinel bit to count zeros only up to bitCount
        zeros = _CTZ(v0 | (1ULL << bitCount));

        v0 >>= zeros;
        vh >>= zeros;
        // Coefficients are two's-complement bit patterns; avoid signed-left-
        // shift UB in the host oracle while preserving CUDA integer semantics.
        *uu = (int64_t)((uint64_t)*uu << zeros);
        *uv = (int64_t)((uint64_t)*uv << zeros);
        bitCount -= zeros;

        if (bitCount == 0)
            break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(y, *uu, *vu);
            SWAP(z, *uv, *vv);
        }

        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;

    }

}
