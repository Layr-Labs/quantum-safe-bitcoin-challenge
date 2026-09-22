// SPDX-License-Identifier: GPL-3.0-only
// Exact cold replay of ambiguous promoted parity windows.
// Public sparse replay description: Saviour1001 8fd91df0.
// Independent two-level sparse bitmap in the existing hit-report allocation.
#pragma once
static_assert(QSB_TREE_N==128 && QSB_S2_THREADS==128 && QSB_RECOVERY_N==128,
              "Bitmap replay uses the promoted 128-leaf geometry");
static_assert(QSB_PARITY_WINDOW && QSB_PARITY_SUM && !QSB_RAW_X &&
              !QSB_TREE_OFFLOAD && !QSB_TREE_OFFLOAD2,
              "Bitmap replay retains the current packed parity path");

__device__ __forceinline__ uint32_t qsb_bitmap_probe(
    const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
    uint64_t mid,top;
    qsb_parity_window_words(mid,top,a,b);
    const uint32_t x7=(uint32_t)mid;
    // Only bit 32 and bits 0..31 of q are used. u64 overflow is harmless.
    const uint64_t q=top+977ULL*(top>>32)+x7+(beta[3]>>32);
    // Unknown carries change q by at most 1958. Exclude the final all-one
    // limb too, so the baseline sum-parity exceptional correction cannot fire.
    // Relative to the promoted window: omitted middle carry <=12, high <=4.
    // Thus delta(q)<=4+977+12=993. Keep all ambiguous cases on the full fallback.
#if QSB_PARITY_HI_WINDOW
    if(x7<0xfffffff3u && (uint32_t)q<0xfffff478u) {
#else
    if(x7!=0xffffffffu && (uint32_t)q<0xfffff859u) {
#endif
        return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(q>>32)^neg)&1u);
    }
    return 2u; // Ambiguous; no hash or nomination before replay.
}


template<bool FULL>
__device__ __forceinline__ uint32_t qsb_bitmap_parity(
    const uint64_t *a,const uint64_t *b,const uint64_t *beta,uint32_t neg) {
    if(FULL) {
        uint64_t raw[4];qsb_packed_raw_mul(raw,a,b);
        return qsb_sum_parity(raw,beta,neg);
    }
    return qsb_bitmap_probe(a,b,beta,neg);
}

template<bool FULL>
__device__ __forceinline__ uint32_t qsb_bitmap_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],s[4];
#if QSB_LAZY_REC
    /* u, v, l, m and sum only feed multiplies and borrow-corrected subtractions, which
     * accept any representative in [0,2^256); only x1/x2 (hashed) and the parity inputs
     * need [0,p). So u and v stay raw and m, sum use the carry-folding lazy add
     * (congruent, [0,2^256); a second carry needs a 2^-223 input, as in the chain). */
    qsb_packed_raw_mul(u,tbar,weighted_inv);
    qsb_packed_raw_mul(v,vbar,root_inv);
#if QSB_NEGATIVE_MAC
    // The checkpoint carries -Y; restore the same l/m slopes and key order.
    _ModAddLazy(l,u,v); _ModSub256(m,u,v); _ModAddLazy(sum,l,m);
#else
    _ModSub256(l,u,v); _ModAddLazy(m,u,v); _ModAddLazy(sum,l,m);
#endif
#else
    qsb_recovery_mul(u,tbar,weighted_inv);
    qsb_recovery_mul(v,vbar,root_inv);
#if QSB_NEGATIVE_MAC
    // The checkpoint carries -Y; restore the same l/m slopes and key order.
    _ModAdd256(l,u,v); _ModSub256(m,u,v); _ModAdd256(sum,l,m);
#else
    _ModSub256(l,u,v); _ModAdd256(m,u,v); _ModAdd256(sum,l,m);
#endif
#endif
#if QSB_RAW_X
    /* P7: x1, x2 stay raw; raw + a < 2p whenever a[3] != 2^64-1, so the one conditional
     * subtraction in _ModAdd256 still yields canonical x (qsb_add_boundary keeps the
     * normalisation for the other case). */
    _ModSub256(t,l,c); qsb_packed_raw_mul(x1,sum,t); qsb_add_boundary(x1,a); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_packed_raw_mul(x2,sum,t); qsb_add_boundary(x2,a); _ModAdd256(x2,x2,a);
#elif !QSB_PARITY_SUM
    _ModSub256(t,l,c); qsb_recovery_mul(x1,sum,t); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_recovery_mul(x2,sum,t); _ModAdd256(x2,x2,a);
#endif
#if QSB_PARITY_SUM
    /* P9: r_i = x_i - a is the canonical product before "+a" (re-derived here with one
     * subtraction-free identity: x_i - a == sum*(l or m - c)), so a - x_i == -r_i and
     * s1 = l*(a-x1) == -(l*r1), s2 = m*(a-x2) == -(m*r2). See qsb_sum_parity. */
    _ModSub256(t,l,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x1,s,a);
#if QSB_PARITY_WINDOW
    const uint32_t parity_u=qsb_bitmap_parity<FULL>(l,s,b,1u);
    if(parity_u&2u)return 4u;
#else
    qsb_packed_raw_mul(u,l,s);
#endif
    _ModSub256(t,m,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x2,s,a);
#if QSB_PARITY_WINDOW
    const uint32_t parity_v=qsb_bitmap_parity<FULL>(m,s,b,0u);
    if(parity_v&2u)return 4u;
#else
    qsb_packed_raw_mul(v,m,s);
#endif
#if QSB_PARITY_WINDOW
    return parity_u|(parity_v<<1);
#else
    return qsb_sum_parity(u,b,1u)|(qsb_sum_parity(v,b,0u)<<1);
#endif
}
#else
    _ModSub256(t,a,x1); qsb_packed_raw_mul(s,l,t); qsb_parity_boundary(s,b);
    uint32_t parity=qsb_difference_parity(s,b);
    _ModSub256(t,a,x2); qsb_packed_raw_mul(s,m,t); qsb_parity_boundary(s,b);
    return parity|(qsb_difference_parity(b,s)<<1);
}
#endif


// Returns true only when the caller must mark this candidate for full replay.
template<bool FAST_TAIL,bool FULL>
__device__ __forceinline__ bool qsb_bitmap_finish_one(
    unsigned idx,int batch_size,int easy_mode,int single_hash,
    const ulonglong2 *saved,const uint64_t *roots,
    uint32_t *d_hit_cnt,uint32_t *d_hit_idx) {
    const size_t root_index=(size_t)idx/QSB_RECOVERY_N;
    uint64_t qy[4],qzzz[4],prod[5];
    size_t i=(size_t)idx,s=(size_t)batch_size;
#if QSB_STREAM2
    ulonglong2 y01=qsb_ld_v2(&saved[0*s+i]),y23=qsb_ld_v2(&saved[1*s+i]);
    ulonglong2 v01=qsb_ld_v2(&saved[2*s+i]),v23=qsb_ld_v2(&saved[3*s+i]);
#else
    ulonglong2 y01=saved[0*s+i],y23=saved[1*s+i];
    ulonglong2 v01=saved[2*s+i],v23=saved[3*s+i];
#endif
    qy[0]=y01.x;qy[1]=y01.y;qy[2]=y23.x;qy[3]=y23.y;
    qzzz[0]=v01.x;qzzz[1]=v01.y;qzzz[2]=v23.x;qzzz[3]=v23.y;
    if((qzzz[0]|qzzz[1]|qzzz[2]|qzzz[3])==0)return false;
    uint64_t weighted_inv[4];
    size_t root_count=((size_t)batch_size+QSB_TREE_N-1)/QSB_TREE_N;
#if QSB_ROOT_V2
    {   /* roots is cudaMalloc'd (256-byte aligned) and indexed in 4-limb (32-byte) records */
        const ulonglong2 *r2=(const ulonglong2 *)roots;
        ulonglong2 a01=r2[2ull*root_index],a23=r2[2ull*root_index+1];
        ulonglong2 b01=r2[2ull*(root_count+root_index)],b23=r2[2ull*(root_count+root_index)+1];
        prod[0]=a01.x;prod[1]=a01.y;prod[2]=a23.x;prod[3]=a23.y;
        weighted_inv[0]=b01.x;weighted_inv[1]=b01.y;weighted_inv[2]=b23.x;weighted_inv[3]=b23.y;
    }
#else
    for(int k=0;k<4;k++)prod[k]=roots[4ull*root_index+k];
    for(int k=0;k<4;k++)weighted_inv[k]=roots[4ull*(root_count+root_index)+k];
#endif
    prod[4]=0;

    uint64_t u2rx[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                      pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t u2ry[4]={pin_u2ry_words[0],pin_u2ry_words[1],
                      pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t recovery_c[4]={pin_recovery_c[0],pin_recovery_c[1],
                            pin_recovery_c[2],pin_recovery_c[3]};
    uint64_t q1x[4],q2x[4];
    uint32_t y_parities = qsb_bitmap_packed_finish<FULL>(
        qy,qzzz,prod,weighted_inv,u2rx,u2ry,recovery_c,q1x,q2x);
    if(y_parities&4u)return true; // Both keys are deferred together.

    /* Check both pubkeys × 2 hashes */
#if QSB_PK_UNROLL
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int ri=0;ri<2;ri++){
        uint64_t sx0=ri ? q2x[0] : q1x[0];
        uint64_t sx1=ri ? q2x[1] : q1x[1];
        uint64_t sx2=ri ? q2x[2] : q1x[2];
        uint64_t sx3=ri ? q2x[3] : q1x[3];
        uint32_t x0=(uint32_t)sx0, x1=(uint32_t)(sx0>>32);
        uint32_t x2=(uint32_t)sx1, x3=(uint32_t)(sx1>>32);
        uint32_t x4=(uint32_t)sx2, x5=(uint32_t)(sx2>>32);
        uint32_t x6=(uint32_t)sx3, x7=(uint32_t)(sx3>>32);
        uint32_t pb[16];
        pb[0]=__byte_perm(x7,0x2+(uint8_t)((y_parities>>ri)&1u),0x4321);
        pb[1]=__byte_perm(x7,x6,0x0765);pb[2]=__byte_perm(x6,x5,0x0765);
        pb[3]=__byte_perm(x5,x4,0x0765);pb[4]=__byte_perm(x4,x3,0x0765);
        pb[5]=__byte_perm(x3,x2,0x0765);pb[6]=__byte_perm(x2,x1,0x0765);
        pb[7]=__byte_perm(x1,x0,0x0765);pb[8]=__byte_perm(x0,0x80,0x0456);
#if QSB_SHA_OPT && QSB_SPARSE_D && QSB_ZEROS_N <= 32
        if (FAST_TAIL) {
            /* ranked gate: only digest word 0 is read */
            if (gpu_bench_valid_h0(_SHA256Pubkey33H0(pb))) {
                uint32_t pos=atomicAdd(d_hit_cnt,1);
                if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30);
                return false;
            }
            continue;
        }
#endif
        uint32_t hs[8];
#if QSB_SPARSE_D
        _SHA256TransformPubkey33(hs,pb);   /* pb[9..14]=0, pb[15]=0x108 folded in */
#else
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        _SHA256Initialize(hs);_SHA256Transform(hs,pb);
#endif
        int vv;
        if (!FAST_TAIL && easy_mode) {
            uint8_t h[32];
            for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
                h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
            vv=gpu_is_der_easy(h,32);
        } else {
            vv=gpu_bench_valid_words(hs);
        }
        if(vv){
            uint32_t pos=atomicAdd(d_hit_cnt,1);
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30);
            return false;
        }
        if (FAST_TAIL || single_hash) continue;  /* Config A: only one hash iteration */
        uint8_t h[32];
        for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        uint8_t pp[64];memset(pp,0,64);memcpy(pp,h,32);pp[32]=0x80;pp[62]=1;pp[63]=0;
        uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
            ((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
        if (!FAST_TAIL && easy_mode) {
            uint8_t h2[32];
            for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
                h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
            vv=gpu_is_der_easy(h2,32);
        } else {
            vv=gpu_bench_valid_words(h2s);
        }
        if(vv){
            uint32_t pos=atomicAdd(d_hit_cnt,1);
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30)|(1u<<31);
            return false;
        }
    }
    return false;
}

// 1 counter + 1024 indices + 31 words of padding. Bitmap begins at 128B alignment.
static constexpr unsigned QSB_BITMAP_REPORT_WORDS=1056u;
static inline size_t qsb_bitmap_report_bytes(unsigned n) {
    const size_t words=((size_t)n+31u)/32u;
    return ((size_t)QSB_BITMAP_REPORT_WORDS+words+(words+31u)/32u)*sizeof(uint32_t);
}
template<bool FAST_TAIL>
__global__ void __launch_bounds__(128,QSB_S2_BLOCKS) qsb_bitmap_finish_hot(
    int n,int easy_mode,int single_hash,const ulonglong2 *saved,
    const uint64_t *roots,uint32_t *hits,uint32_t *hit_idx) {
    const unsigned idx=blockIdx.x*128u+threadIdx.x;
    if(idx>=(unsigned)n)return;
    if(qsb_bitmap_finish_one<FAST_TAIL,false>(idx,n,easy_mode,single_hash,
                                            saved,roots,hits,hit_idx)) {
        const unsigned words=((unsigned)n+31u)/32u;
        atomicOr(hits+QSB_BITMAP_REPORT_WORDS+(idx>>5),1u<<(idx&31u));
        atomicOr(hits+QSB_BITMAP_REPORT_WORDS+words+(idx>>10),1u<<((idx>>5)&31u));
    }
}

template<bool FAST_TAIL>
__global__ void __launch_bounds__(128,4) qsb_bitmap_finish_replay(
    int n,int easy_mode,int single_hash,const ulonglong2 *saved,
    const uint64_t *roots,uint32_t *hits,uint32_t *hit_idx) {
    const unsigned words=((unsigned)n+31u)/32u;
    const unsigned group=blockIdx.x*128u+threadIdx.x;
    if(group>=(words+31u)/32u)return;
    unsigned nonempty=hits[QSB_BITMAP_REPORT_WORDS+words+group];
    while(nonempty) {
        const unsigned word=(group<<5)+(unsigned)(__ffs(nonempty)-1);
        unsigned bits=hits[QSB_BITMAP_REPORT_WORDS+word];
        while(bits) {
            const unsigned idx=(word<<5)+(unsigned)(__ffs(bits)-1);
            if(idx<(unsigned)n)
                (void)qsb_bitmap_finish_one<FAST_TAIL,true>(idx,n,easy_mode,single_hash,
                                                          saved,roots,hits,hit_idx);
            bits&=bits-1u;
        }
        nonempty&=nonempty-1u;
    }
}

// Integration must call this AFTER root finish and BEFORE copies/completion.
// Each slot allocates qsb_bitmap_report_bytes(max_batch) at hits and sets
// hit_idx=hits+1. Expand the PRE-EXISTING pre-prepare report memset to
// qsb_bitmap_report_bytes(batch_size). Do not clear it between hot/replay.
// This function does not clear the report itself: that would erase hits.
template<bool FAST_TAIL>
static cudaError_t qsb_bitmap_launch_finish(
    int n,int easy_mode,int single_hash,const ulonglong2 *saved,
    const uint64_t *roots,uint32_t *hits,uint32_t *hit_idx,
    cudaStream_t stream) {
    if(n<=0)return cudaSuccess;
    qsb_bitmap_finish_hot<FAST_TAIL><<<((unsigned)n+127u)/128u,128,0,stream>>>(
        n,easy_mode,single_hash,saved,roots,hits,hit_idx);
    cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
    const unsigned words=((unsigned)n+31u)/32u;
    qsb_bitmap_finish_replay<FAST_TAIL><<<(((words+31u)/32u)+127u)/128u,128,0,stream>>>(
        n,easy_mode,single_hash,saved,roots,hits,hit_idx);
    return cudaGetLastError();
}

// Query once at startup, outside the candidate loop. This observes resources;
// it does not select a kernel, impose a register cap, or run a tuning workload.
static cudaError_t qsb_bitmap_report_resources() {
    cudaFuncAttributes hot={},cold={};
    cudaError_t err=cudaFuncGetAttributes(&hot,qsb_bitmap_finish_hot<true>);
    if(err!=cudaSuccess)return err;
    err=cudaFuncGetAttributes(&cold,qsb_bitmap_finish_replay<true>);
    if(err!=cudaSuccess)return err;
    fprintf(stderr,"QSB parity replay resources: hot regs=%d local=%zu shared=%zu; "
                   "cold regs=%d local=%zu shared=%zu; report=%zu bytes/slot\n",
            hot.numRegs,hot.localSizeBytes,hot.sharedSizeBytes,
            cold.numRegs,cold.localSizeBytes,cold.sharedSizeBytes,
            qsb_bitmap_report_bytes(QSB_BATCH));
    return cudaSuccess;
}
