#define main integrated_candidate_main
#include "candidate/pinning.cu"
#undef main

template<bool FAST_TAIL>
__global__ void __launch_bounds__(256, 2) kernel_pinning_fused(
    const uint32_t *__restrict__ d_midstate,
    const uint8_t *__restrict__ d_suffix,    /* suffix template */
    int suffix_len,             /* total suffix including lt+sighash */
    int seq_offset,             /* offset of sequence in suffix */
    int lt_offset,              /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,         /* current sequence value */
    uint32_t start_lt,          /* starting locktime for this batch */
    const uint64_t *__restrict__ d_neg_r_inv,
    const uint64_t *__restrict__ d_u2rx, const uint64_t *__restrict__ d_u2ry,
    const uint64_t *__restrict__ d_neg2u2rx, const uint64_t *__restrict__ d_neg2u2ry,
    const uint8_t *__restrict__ d_gt,
    uint32_t *__restrict__ d_hit_cnt, uint32_t *__restrict__ d_hit_idx,
    int batch_size, int easy_mode, int single_hash, uint32_t group_slot,
    ulonglong2 *__restrict__ saved, uint64_t *__restrict__ roots,
    uint64_t *__restrict__ tree
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x * blockDim.x >= batch_size) return;
    int active = idx < batch_size;
    uint32_t lt = start_lt + (uint32_t)(active ? idx : 0);

    uint64_t qx[4], qy[4], qzz[4], qzzz[4], prod[5];
    uint32_t state[8];
    if (FAST_TAIL) {
        // This specialization is selected only for single_hash, normal mode.
        easy_mode = 0;
        single_hash = 1;
        #pragma unroll
        for (int i=0;i<8;i++) state[i]=d_midstate[i];
        uint32_t blk[16] = {
            pin_tail_words[0] | (lt & 0xffu),
            ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1],
            pin_tail_words[2],
            0,0,0,0,0,0,0,0,0,0,0,0,9995u*8u
        };
        _SHA256Transform(state,blk);
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
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000u;
    #pragma unroll
    for(int i=9;i<15;i++) b2[i]=0;
    b2[15]=256;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);

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
    _FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);

    /* Recover P+R and P-R together with one shared denominator inverse. The
     * prepare-only xR copy dies before the collective; reload R afterward so
     * its eight limbs do not lengthen the inverse's already pressured state. */
    {
        uint64_t prep_xR[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                             pin_u2rx_words[2],pin_u2rx_words[3]};
        qsb_xyzz_finish_prepare(qx,qzz,prep_xR,prod);
    }
    bool usable = active && ((prod[0] | prod[1] | prod[2] | prod[3]) != 0);
    /* Preserve exactly four fields across the kernel boundary.  The finish
     * needs C=ZZ*d^2 and W=ZZ^2*d, but no longer needs d or ZZ separately. */
    if(usable){
        _ModSqr(qx,qx);
        _ModMult(qx,qzz);        /* qx becomes C */
    }
    Load256(qzz,prod);           /* qzz becomes W */
    if (!usable) {
        prod[0]=1; prod[1]=prod[2]=prod[3]=prod[4]=0;
    }
    qsb_block_inverse(prod);
    if (!usable) return;
    uint64_t u2rx[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                      pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t u2ry[4]={pin_u2ry_words[0],pin_u2ry_words[1],
                      pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t q1x[4],q2x[4];
    uint32_t y_parities = qsb_xyzz_finish_precomputed(
        qx,qy,qzz,qzzz,prod,u2rx,u2ry,
        q1x,q2x);

    /* Check both pubkeys × 2 hashes */
    #pragma unroll
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
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
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
            if(pos<1024){
                if(FAST_TAIL)
                    d_hit_idx[pos]=(lt & 0x7FFFFFFFu)|(ri<<31);
                else
                    d_hit_idx[pos]=((uint32_t)idx)|(group_slot<<24)|(ri<<30);
            }
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
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(group_slot<<24)|(ri<<30)|(1u<<31);
            return;
        }
    }
}

void instantiate_fused_true(const uint32_t *mid, const uint8_t *suffix, int suffix_len,
    int seq_offset, int lt_offset, int total_len, uint32_t seq, uint32_t lt,
    const uint64_t *nri, const uint64_t *rx, const uint64_t *ry,
    const uint64_t *n2x, const uint64_t *n2y, const uint8_t *table,
    uint32_t *hit_count, uint32_t *hits, int count, int easy, int single,
    uint32_t slot) {
    kernel_pinning_fused<true><<<(count+255)/256,256>>>(mid,suffix,suffix_len,seq_offset,lt_offset,total_len,seq,lt,nri,rx,ry,n2x,n2y,table,hit_count,hits,count,easy,single,slot,nullptr,nullptr,nullptr);
}
void instantiate_fused_false(const uint32_t *mid, const uint8_t *suffix, int suffix_len,
    int seq_offset, int lt_offset, int total_len, uint32_t seq, uint32_t lt,
    const uint64_t *nri, const uint64_t *rx, const uint64_t *ry,
    const uint64_t *n2x, const uint64_t *n2y, const uint8_t *table,
    uint32_t *hit_count, uint32_t *hits, int count, int easy, int single,
    uint32_t slot) {
    kernel_pinning_fused<false><<<(count+255)/256,256>>>(mid,suffix,suffix_len,seq_offset,lt_offset,total_len,seq,lt,nri,rx,ry,n2x,n2y,table,hit_count,hits,count,easy,single,slot,nullptr,nullptr,nullptr);
}

int main(){return 0;} // Compile-only probe: never launches a GPU kernel.
