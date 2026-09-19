// SPDX-License-Identifier: GPL-3.0-only
// SHA bodies preserved from promoted 9fab500; R16-W5 scheduling is new.
#pragma once
#ifndef QSB_W5_COHORTS
#define QSB_W5_COHORTS 128
#endif
static_assert(QSB_W5_COHORTS>0,"positive cohort count");

template<bool FAST_TAIL>
__global__ void __launch_bounds__(256,2) w5_produce(
    const uint32_t *d_midstate,const uint8_t *d_suffix,int suffix_len,
    int seq_offset,int lt_offset,int total_preimage_len,uint32_t seq_value,
    uint32_t start_lt,int batch_size,int easy_mode,int single_hash,ulonglong2 *saved) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=batch_size)return;
    uint32_t lt=start_lt+(uint32_t)idx;
    uint32_t state[8];
    if (FAST_TAIL) {
        // This specialization is selected only for single_hash, normal mode.
        easy_mode = 0;
        single_hash = 1;
        #pragma unroll
        for (int i=0;i<8;i++) state[i]=d_midstate[i];
#if QSB_SPARSE_TAIL
        /* W[0..2] live locktime-patched words; W[3..14]=0; W[15]=79960. */
        uint32_t w0 = pin_tail_words[0] | (lt & 0xffu);
        uint32_t w1 = ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1];
        uint32_t w2 = pin_tail_words[2];
        _SHA256TransformFastTail11(state, w0, w1, w2);
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
#if QSB_SPARSE_D
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

    qsb_st_v2(saved+idx,z[0],z[1]);
    qsb_st_v2(saved+(size_t)batch_size+idx,z[2],z[3]);
}

template<bool FAST_TAIL>
__global__ void __launch_bounds__(256,2) w5_hash_finish(
    ulonglong2 *saved,int batch_size,int easy_mode,int single_hash,
    uint32_t *d_hit_cnt,uint32_t *d_hit_idx) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=batch_size)return;
    size_t s=(size_t)batch_size;
    uint32_t status=(uint32_t)qsb_ld_v2(saved+4*s+idx).x;
    uint32_t y_parities=status&3u,valid=status>>2;
    if(!valid)return;
    ulonglong2 a=qsb_ld_v2(saved+idx),b=qsb_ld_v2(saved+s+idx);
    ulonglong2 c=qsb_ld_v2(saved+2*s+idx),d=qsb_ld_v2(saved+3*s+idx);
    uint64_t q1x[4]={a.x,a.y,b.x,b.y},q2x[4]={c.x,c.y,d.x,d.y};
    /* Check both pubkeys × 2 hashes */
#if QSB_PK_UNROLL
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int ri=0;ri<2;ri++){
        if(!(valid & (1u<<ri))) continue;
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
