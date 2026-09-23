// SPDX-License-Identifier: GPL-3.0-only
// Resident affine workers for the real pinning search. Include after pinning.cu's
// field, SHA, signed-recode and table helpers. The host owns cooperative launch,
// zeroed mailboxes/output masks, exact nomination checks and exception fallback.
#pragma once
#include "research_affine/inverse_tree.cuh"
#include "research_affine/inverse_service.cuh"
#include "research_affine/shifted_tail.cuh"

#if QSB_ISO_XR || QSB_YOFF
#error "Affine search requires canonical original-curve tables (ISO_XR=YOFF=0)"
#endif
#if !QSB_AFFINE_TREE_EXACT
#error "Affine search requires carry-complete shared inverse trees"
#endif

constexpr unsigned QSB_AFFINE_SEARCH_N=128;
struct QsbAffineAudit {uint64_t x[2][4];uint32_t parities,usable;};

// Deliberately complete reference arithmetic for the first ranked GPU test.
// Specialized squaring/parity can be measured separately after correctness.
struct QsbAffineSearchField {
    __device__ __forceinline__ static void sub(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]) {
        uint64_t r0,r1,r2,r3;
        asm volatile("{ .reg .u64 m,k;\n\t"
          "sub.cc.u64 %0,%4,%8; subc.cc.u64 %1,%5,%9;\n\t"
          "subc.cc.u64 %2,%6,%10; subc.cc.u64 %3,%7,%11;\n\t"
          "subc.u64 m,0,0; and.b64 k,m,0xfffffffefffffc2f;\n\t"
          "add.cc.u64 %0,%0,k; addc.cc.u64 %1,%1,m;\n\t"
          "addc.cc.u64 %2,%2,m; addc.u64 %3,%3,m; }"
          :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
          :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
           "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
        out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
    }
    __device__ __forceinline__ static void neg(uint64_t out[4],const uint64_t a[4]) {
        const uint64_t zero[4]={0,0,0,0};sub(out,zero,a);
    }
    __device__ __forceinline__ static void mul(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]) {
        uint64_t aa[5],bb[5],r[5];
        #pragma unroll
        for(unsigned k=0;k<4;k++){aa[k]=a[k];bb[k]=b[k];}
        aa[4]=bb[4]=0;qsb_field_mul(r,aa,bb);qsb_field_normalize(r);
        #pragma unroll
        for(unsigned k=0;k<4;k++)out[k]=r[k];
    }
    __device__ __forceinline__ static void sqr(uint64_t out[4],const uint64_t a[4]) {mul(out,a,a);}
    __device__ __forceinline__ static uint32_t parity_product(
        const uint64_t a[4],const uint64_t b[4],const uint64_t beta[4],uint32_t negative) {
        uint64_t product[4],minus_beta[4];
        mul(product,a,b);neg(minus_beta,beta);sub(product,product,minus_beta);
        if(negative)neg(product,product);
        return uint32_t(product[0]&1u);
    }
};

struct QsbAffineSearchInvert {
    __device__ __forceinline__ void operator()(uint64_t out[4],const uint64_t in[4]) const {
        uint64_t t[5]={in[0],in[1],in[2],in[3],0};
        _ModInv(t);qsb_field_normalize(t);
        #pragma unroll
        for(unsigned k=0;k<4;k++)out[k]=t[k];
    }
};

__device__ __noinline__ void qsb_affine_search_service(
    qsb_inverse_service::Slot* slots,unsigned workers,unsigned service_index) {
    qsb_inverse_service::service(slots,workers,service_index,QsbAffineSearchInvert{});
}

// Real ranked 9995-byte message geometry. The host validates FAST_TAIL and
// refreshes tp.mid and pin_tail_words for each sequence exactly as before.
// Use the actual locktime, not blockIdx: some blocks are inversion services and
// each worker processes many tiles in a single launch.
__device__ __forceinline__ void qsb_affine_search_hash_scalar(
    uint64_t z[4],uint32_t lt,const qsb_tail_pre& tp) {
    uint32_t state[8],digest[8];
    #pragma unroll
    for(unsigned k=0;k<8;k++)state[k]=tp.mid[k];
    const uint32_t w0=pin_tail_words[0]|(lt&0xffu);
    const uint32_t w1=((lt&0xff00u)<<16)|(lt&0xff0000u)|
                      ((lt>>16)&0xff00u)|pin_tail_words[1];
    const uint32_t w2=pin_tail_words[2];
#if QSB_SHA_OPT
    _SHA256TransformFastTail11Q(state,w0,w1,w2,tp);
    _SHA256TransformDigest32Q(digest,state);
#else
    _SHA256TransformFastTail11P(state,w0,w1,w2,tp);
    _SHA256TransformDigest32(digest,state);
#endif
    z[0]=(uint64_t(digest[6])<<32)|digest[7];
    z[1]=(uint64_t(digest[4])<<32)|digest[5];
    z[2]=(uint64_t(digest[2])<<32)|digest[3];
    z[3]=(uint64_t(digest[0])<<32)|digest[1];
}

// Same exact signed2k-n extraction as qsb_decode_to_shared, but the codes own
// 7.5 KiB of shared storage while the separate 12 KiB inverse tree is live.
__device__ __forceinline__ void qsb_affine_search_recode(
    const uint64_t z[4],uint32_t (*codes)[QSB_AFFINE_SEARCH_N]) {
    uint64_t m[4];int negative;qsb_signed_recode_setup(z,m,&negative);
    #pragma unroll
    for(unsigned c=0;c<GT_CHUNKS;c++) {
        const unsigned pos=c==0?1u:17u*c+2u;
        const unsigned j=pos/64u,shift=pos%64u;
        uint64_t value=m[j]>>shift;
        if(j<3 && shift>46u)value|=m[j+1]<<(64u-shift);
        const unsigned bits=c==0?18u:17u;
        const uint32_t f=uint32_t(value)&((1u<<bits)-1u);
        const int32_t mask=c==GT_CHUNKS-1?-negative:int32_t(f>>(bits-1u))-1;
        const uint32_t index=(f^uint32_t(mask))&((1u<<(bits-1u))-1u);
        codes[c][threadIdx.x]=index|(uint32_t(mask<0)<<31);
    }
}

__device__ __forceinline__ QsbShiftedTailPoint qsb_affine_search_load(
    const uint8_t* table,unsigned window,uint32_t code) {
    QsbShiftedTailPoint p;
    gt_load_signed_flat_m(table,gt_offset(window),code&0x1ffffu,
                         0ULL-uint64_t(code>>31),p.x,p.y);
    return p;
}

// Hash the actual compressed 33-byte secp256k1 public key. Test each recovery
// arm independently so a candidate with two matching keys retains both bits.
__device__ __forceinline__ bool qsb_affine_search_pubkey_matches(
    const uint64_t x[4],uint32_t parity) {
    const uint32_t x0=uint32_t(x[0]),x1=uint32_t(x[0]>>32);
    const uint32_t x2=uint32_t(x[1]),x3=uint32_t(x[1]>>32);
    const uint32_t x4=uint32_t(x[2]),x5=uint32_t(x[2]>>32);
    const uint32_t x6=uint32_t(x[3]),x7=uint32_t(x[3]>>32);
    uint32_t pb[16];
    pb[0]=__byte_perm(x7,2u+(parity&1u),0x4321);
    pb[1]=__byte_perm(x7,x6,0x0765);pb[2]=__byte_perm(x6,x5,0x0765);
    pb[3]=__byte_perm(x5,x4,0x0765);pb[4]=__byte_perm(x4,x3,0x0765);
    pb[5]=__byte_perm(x3,x2,0x0765);pb[6]=__byte_perm(x2,x1,0x0765);
    pb[7]=__byte_perm(x1,x0,0x0765);pb[8]=__byte_perm(x0,0x80,0x0456);
#if QSB_SHA_OPT && QSB_ZEROS_N <= 32
    return gpu_bench_valid_h0(_SHA256Pubkey33H0(pb));
#else
    uint32_t digest[8];_SHA256TransformPubkey33(digest,pb);
    return gpu_bench_valid_words(digest);
#endif
}

// Only x coordinates and infinity sentinels are needed before inversion.
// Reload the complete final points after the reply to bound register pressure.
__device__ __forceinline__ bool qsb_affine_search_tail_denominator(
    uint64_t denominator[4],const QsbShiftedTailPoint& prefix,
    const QsbShiftedTailRecord* tails,uint32_t index) {
    const QsbShiftedTailRecord& record=tails[index];
    if(qsb_shifted_tail_infinity(prefix)||qsb_shifted_tail_infinity(record.arm[0])||
       qsb_shifted_tail_infinity(record.arm[1]))return false;
    uint64_t d0[4],d1[4];
    QsbAffineSearchField::sub(d0,record.arm[0].x,prefix.x);
    QsbAffineSearchField::sub(d1,record.arm[1].x,prefix.x);
    if(qsb_shifted_tail_zero(d0)||qsb_shifted_tail_zero(d1))return false;
    QsbAffineSearchField::mul(denominator,d0,d1);
    return true;
}

template<bool AUDIT>
__device__ __forceinline__ uint32_t qsb_affine_search_finish_matches(
    const uint64_t inverse[4],const QsbShiftedTailPoint& prefix,
    const QsbShiftedTailRecord* tails,uint32_t index,uint32_t negative,
    QsbAffineAudit* audit) {
    uint32_t matches=0,parities=0;
    #pragma unroll
    for(unsigned arm=0;arm<2;arm++) {
        QsbShiftedTailPoint tail=tails[index].arm[arm^negative];
        if(negative)QsbAffineSearchField::neg(tail.y,tail.y);
        uint64_t other_delta[4],numerator[4],slope[4],x[4],offset[4];
        QsbAffineSearchField::sub(other_delta,tails[index].arm[(arm^1u)^negative].x,prefix.x);
        QsbAffineSearchField::sub(numerator,tail.y,prefix.y);
        QsbAffineSearchField::mul(slope,numerator,other_delta);
        QsbAffineSearchField::mul(slope,slope,inverse);
        QsbAffineSearchField::sqr(x,slope);
        QsbAffineSearchField::sub(x,x,prefix.x);QsbAffineSearchField::sub(x,x,tail.x);
        QsbAffineSearchField::sub(offset,x,tail.x);
        const uint32_t parity=QsbAffineSearchField::parity_product(slope,offset,tail.y,1u);
        if(AUDIT) {
            #pragma unroll
            for(unsigned k=0;k<4;k++)audit->x[arm][k]=x[k];
            parities|=parity<<arm;
        }
        if(qsb_affine_search_pubkey_matches(x,parity))matches|=1u<<arm;
    }
    if(AUDIT){audit->parities=parities;audit->usable=1;}
    return matches;
}

// Launch exactly service_blocks+workers CTAs, cooperatively, using the runtime
// occupancy limit for this very kernel. Every service lane remains alive until
// its assigned worker stops; a worker never skips a collective for a bad lane.
// hit_mask: ceil(batch_size/16) uint32_t, two bits per candidate (recid0,recid1).
// exception_mask: ceil(batch_size/32) uint32_t, one bit per candidate.
// Both masks and every mailbox must be zeroed before each independent launch.
template<bool AUDIT=false>
__global__ __launch_bounds__(QSB_AFFINE_SEARCH_N,4)
void qsb_affine_search(const uint8_t* table,const QsbShiftedTailRecord* tails,
    qsb_inverse_service::Slot* slots,uint32_t* hit_mask,uint32_t* exception_mask,
    uint32_t batch_size,uint32_t start_lt,qsb_tail_pre tp,
    unsigned service_blocks,unsigned workers,QsbAffineAudit* audit) {
    if(blockIdx.x<service_blocks) {
        qsb_affine_search_service(slots,workers,blockIdx.x*QSB_AFFINE_SEARCH_N+threadIdx.x);
        return;
    }
    const unsigned worker=blockIdx.x-service_blocks;
    __shared__ uint64_t products[4][2*QSB_AFFINE_SEARCH_N],inverses[4][QSB_AFFINE_SEARCH_N];
    __shared__ uint32_t codes[GT_CHUNKS][QSB_AFFINE_SEARCH_N];
    const uint64_t stride=uint64_t(workers)*QSB_AFFINE_SEARCH_N;
    uint32_t epoch=0;
    for(uint64_t base=uint64_t(worker)*QSB_AFFINE_SEARCH_N;base<batch_size;base+=stride) {
        const uint32_t idx=uint32_t(base)+threadIdx.x;
        const bool active=idx<batch_size;
        uint64_t z[4];qsb_affine_search_hash_scalar(z,start_lt+idx,tp);
        qsb_affine_search_recode(z,codes);
        QsbShiftedTailPoint point=qsb_affine_search_load(table,0,codes[0][threadIdx.x]);
        bool valid=active&&!qsb_shifted_tail_infinity(point);
        #pragma unroll 1
        for(unsigned window=1;window<14;window++) {
            const QsbShiftedTailPoint addend=qsb_affine_search_load(table,window,codes[window][threadIdx.x]);
            uint64_t delta[4],numerator[4],root[4],inverse[4];
            QsbAffineSearchField::sub(delta,addend.x,point.x);
            QsbAffineSearchField::sub(numerator,addend.y,point.y);
            valid=valid&&!qsb_shifted_tail_infinity(addend)&&!qsb_shifted_tail_zero(delta);
            if(!valid){delta[0]=1;delta[1]=delta[2]=delta[3]=0;}
            qsb_affine_tree_prepare_shared<QSB_AFFINE_SEARCH_N>(delta,root,products);
            ++epoch;
            if(threadIdx.x==0) {
                qsb_inverse_service::worker_publish(slots[worker],epoch,root);
                qsb_inverse_service::worker_wait(slots[worker],epoch,root);
            }
            qsb_affine_tree_finish_shared<QSB_AFFINE_SEARCH_N>(inverse,root,products,inverses);
            uint64_t slope[4],next_x[4],offset[4];
            QsbAffineSearchField::mul(slope,numerator,inverse);
            QsbAffineSearchField::sqr(next_x,slope);
            QsbAffineSearchField::sub(next_x,next_x,point.x);
            QsbAffineSearchField::sub(next_x,next_x,addend.x);
            QsbAffineSearchField::sub(offset,addend.x,next_x);
            QsbAffineSearchField::mul(point.y,slope,offset);
            QsbAffineSearchField::sub(point.y,point.y,addend.y);
            #pragma unroll
            for(unsigned k=0;k<4;k++)point.x[k]=next_x[k];
            __syncthreads(); // no inverse-tree leaf reader survives into prepare
        }
        const uint32_t tail_code=codes[14][threadIdx.x];
        const uint32_t tail_index=tail_code&0xffffu,negative=tail_code>>31;
        uint64_t denominator[4],root[4],inverse[4];
        const bool tail_ok=qsb_affine_search_tail_denominator(denominator,point,tails,tail_index);
        valid=valid&&tail_ok;
        if(!valid){denominator[0]=1;denominator[1]=denominator[2]=denominator[3]=0;}
        qsb_affine_tree_prepare_shared<QSB_AFFINE_SEARCH_N>(denominator,root,products);
        ++epoch;
        if(threadIdx.x==0) {
            qsb_inverse_service::worker_publish(slots[worker],epoch,root);
            qsb_inverse_service::worker_wait(slots[worker],epoch,root);
        }
        qsb_affine_tree_finish_shared<QSB_AFFINE_SEARCH_N>(inverse,root,products,inverses);
        if(AUDIT&&active)audit[idx].usable=0;
        if(valid) {
            const uint32_t matches=qsb_affine_search_finish_matches<AUDIT>(
                inverse,point,tails,tail_index,negative,AUDIT?audit+idx:nullptr);
            if(matches)atomicOr(&hit_mask[idx>>4],matches<<((idx&15u)*2u));
        } else if(active) {
            atomicOr(&exception_mask[idx>>5],1u<<(idx&31u));
        }
        __syncthreads(); // all scratch and digit readers finish before next tile
    }
    if(threadIdx.x==0)qsb_inverse_service::worker_stop(slots[worker]);
}
