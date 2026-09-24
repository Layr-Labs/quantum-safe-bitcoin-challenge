// The pipeline body is deliberately included twice when the register-cap
// experiment is enabled. Each instantiation keeps its own kernel annotation.
// Define QSB_PIPELINE_NAME and QSB_PIPELINE_ATTR before including this file.
// No include guard: two different names are required for independent resources.
template<bool FAST_TAIL, int STAGE>
__global__ void QSB_PIPELINE_ATTR QSB_PIPELINE_NAME(
    const uint32_t *d_midstate,
    const uint8_t *d_suffix,    /* suffix template */
    int suffix_len,             /* total suffix including lt+sighash */
    int seq_offset,             /* offset of sequence in suffix */
    int lt_offset,              /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,         /* current sequence value */
    uint32_t start_lt,          /* starting locktime for this batch */
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gt,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode, int single_hash,
    ulonglong2 *saved, uint64_t *roots, uint64_t *tree, qsb_tail_pre tp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x * blockDim.x >= batch_size) return;
    int active = idx < batch_size;
    uint32_t lt = start_lt + (uint32_t)(active ? idx : 0);

    uint64_t qx[4], qy[4], qzz[4], qzzz[4], prod[5];
    if (STAGE==0) {
    uint32_t state[8];
    if (FAST_TAIL) {
        // This specialization is selected only for single_hash, normal mode.
        easy_mode = 0;
        single_hash = 1;
#if QSB_TAIL_PRE
        #pragma unroll
        for (int i=0;i<8;i++) state[i]=tp.mid[i];
#else
        #pragma unroll
        for (int i=0;i<8;i++) state[i]=d_midstate[i];
#endif
#if QSB_SPARSE_TAIL
        /* W[0..2] live locktime-patched words; W[3..14]=0; W[15]=79960. */
#if QSB_SHA_SMEM_W1
        __shared__ uint4 sh_w1[2];
        qsb_tail_w1_block(start_lt, tp, sh_w1);
        uint4 sa = sh_w1[0], sb = sh_w1[1];
        uint32_t b0 = (uint32_t)(((blockIdx.x & 1u) << 7) | threadIdx.x);
        uint32_t w0 = pin_tail_words[0] | b0;
        uint32_t w1 = sa.x;
        (void)lt; (void)w1;
#else
        uint32_t w0, w1;
        qsb_tail_message(start_lt, lt, w0, w1);
#endif
        uint32_t w2 = pin_tail_words[2];
#if QSB_TAIL_PRE && QSB_SHA_OPT && QSB_SHA_SMEM_W1 && QSB_TAIL_TAB
        _SHA256TransformFastTail11ST(state, w0, w2, tp, sa, sb, __ldg(&pin_tail_tab[b0]));
#elif QSB_TAIL_PRE && QSB_SHA_OPT && QSB_SHA_SMEM_W1
        _SHA256TransformFastTail11S(state, w0, w2, tp, sa, sb);
#elif QSB_TAIL_PRE && QSB_SHA_OPT
        _SHA256TransformFastTail11Q(state, w0, w1, w2, tp);
#elif QSB_TAIL_PRE
        _SHA256TransformFastTail11P(state, w0, w1, w2, tp);
#else
        _SHA256TransformFastTail11(state, w0, w1, w2);
#endif
#else
        uint32_t blk[16] = {
            pin_tail_words[0] | (lt & 0xffu),
            ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1],
            pin_tail_words[2],
            0,0,0,0,0,0,0,0,0,0,0,0,9995u*8u
        };
        _SHA256Transform(state,blk);
#endif
    } else {
        /* Copy suffix, set sequence + locktime */
        uint8_t buf[192];
        for(int i=0;i<suffix_len;i++) buf[i]=d_suffix[i];
        buf[seq_offset]=(seq_value)&0xFF; buf[seq_offset+1]=(seq_value>>8)&0xFF;
        buf[seq_offset+2]=(seq_value>>16)&0xFF; buf[seq_offset+3]=(seq_value>>24)&0xFF;
        buf[lt_offset]=(lt)&0xFF; buf[lt_offset+1]=(lt>>8)&0xFF;
        buf[lt_offset+2]=(lt>>16)&0xFF; buf[lt_offset+3]=(lt>>24)&0xFF;

        /* SHA-256 padding */
        buf[suffix_len]=0x80;
        for(int i=suffix_len+1;i<192;i++) buf[i]=0;
        int nblk=(suffix_len<56)?1:2;
        uint64_t bit_len=(uint64_t)total_preimage_len*8;
        int last=nblk*64-8;
        buf[last]=(bit_len>>56)&0xFF;buf[last+1]=(bit_len>>48)&0xFF;
        buf[last+2]=(bit_len>>40)&0xFF;buf[last+3]=(bit_len>>32)&0xFF;
        buf[last+4]=(bit_len>>24)&0xFF;buf[last+5]=(bit_len>>16)&0xFF;
        buf[last+6]=(bit_len>>8)&0xFF;buf[last+7]=bit_len&0xFF;

        for(int i=0;i<8;i++) state[i]=d_midstate[i];
        for(int b=0;b<nblk;b++){
            uint32_t blk[16]; for(int i=0;i<16;i++)
                blk[i]=((uint32_t)buf[b*64+i*4]<<24)|((uint32_t)buf[b*64+i*4+1]<<16)|
                       ((uint32_t)buf[b*64+i*4+2]<<8)|(uint32_t)buf[b*64+i*4+3];
            _SHA256Transform(state,blk);
        }

    }

    /* Second SHA-256: the first digest is already in big-endian words. */
    uint32_t s2[8];
#if QSB_SPARSE_D && QSB_SHA_OPT
    _SHA256TransformDigest32Q(s2, state);
#elif QSB_SPARSE_D
    _SHA256TransformDigest32(s2, state);
#else
    {
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000u;
    #pragma unroll
    for(int i=9;i<15;i++) b2[i]=0;
    b2[15]=256;
    const uint32_t iv[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                          0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    #pragma unroll
    for(int i=0;i<8;i++) s2[i]=iv[i];
    _SHA256Transform(s2,b2);
    }
#endif

    /* Scalar from the SHA-256 state words, in little-endian limbs. */
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    /* neg_r_inv is folded into fixed base A = neg_r_inv*G. Recoding z
     * directly yields z*A = (neg_r_inv*z mod n)*G without a per-candidate
     * scalar multiplication. */
    /* u1*G as raw XYZZ via the signed 64 MiB A-table. */
    _FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt,qsb_prepare_scratch());

    /* Recover P+R and P-R together with one shared denominator inverse. The
     * prepare-only xR copy dies before the collective; reload R afterward so
     * its eight limbs do not lengthen the inverse's already pressured state. */
    {
        uint64_t prep_xR[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                             pin_u2rx_words[2],pin_u2rx_words[3]};
        qsb_recovery_denominator(qx,qzz,qy,qzzz,prep_xR,prod);
    }
    bool usable = active && ((prod[0] | prod[1] | prod[2] | prod[3]) != 0);
    if(!usable){prod[0]=1;prod[1]=prod[2]=prod[3]=prod[4]=0;}
    qsb_packed_prepare(prod,qzz,qy,qzzz,usable,active,batch_size,saved,roots);
    (void)tree;
    return;
    } else {

    if(!active)return;
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
    if((qzzz[0]|qzzz[1]|qzzz[2]|qzzz[3])==0)return;
    uint64_t weighted_inv[4];
    size_t root_count=((size_t)batch_size+QSB_TREE_N-1)/QSB_TREE_N;
#if QSB_ROOT_V2
    {   /* roots is cudaMalloc'd (256-byte aligned) and indexed in 4-limb (32-byte) records */
        const ulonglong2 *r2=(const ulonglong2 *)roots;
        ulonglong2 a01=r2[2ull*blockIdx.x],a23=r2[2ull*blockIdx.x+1];
        ulonglong2 b01=r2[2ull*(root_count+blockIdx.x)],b23=r2[2ull*(root_count+blockIdx.x)+1];
        prod[0]=a01.x;prod[1]=a01.y;prod[2]=a23.x;prod[3]=a23.y;
        weighted_inv[0]=b01.x;weighted_inv[1]=b01.y;weighted_inv[2]=b23.x;weighted_inv[3]=b23.y;
    }
#else
    for(int k=0;k<4;k++)prod[k]=roots[4ull*blockIdx.x+k];
    for(int k=0;k<4;k++)weighted_inv[k]=roots[4ull*(root_count+blockIdx.x)+k];
#endif
    prod[4]=0;
    (void)tree;
    uint64_t u2rx[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                      pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t u2ry[4]={pin_u2ry_words[0],pin_u2ry_words[1],
                      pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t recovery_c[4]={pin_recovery_c[0],pin_recovery_c[1],
                            pin_recovery_c[2],pin_recovery_c[3]};
    uint64_t q1x[4],q2x[4];
    uint32_t y_parities = qsb_packed_finish(
        qy,qzzz,prod,weighted_inv,u2rx,u2ry,recovery_c,q1x,q2x);

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
                return;
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
            return;
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
            return;
        }
    }
    }
}
