#pragma once
/* QSB_ROOT_COMBINE (tree.cu switch block): pairwise cross-block root combining. PROTOTYPE, default off.
 *
 * Every role branch and loop exit is a warp vote on a lane-0 value, the atomics are release/acquire, polls are relaxed
 * loads, the partner root is loaded by lane 0 and shuffled. In particular:
 *  - exact combine products: qsb_field_mul_raw + qsb_field_normalize for the three pair products (not
 *    QSB_TREE_MUL's filter multiply with its rare-carry class), so a combined inverse is the same canonical
 *    element zi_inverse_limbs returns for the block's own root (inv(u)/R_self = inv(u)/(R_self R_partner) *
 *    R_partner), and the fixed-work hit set must equal the base's exactly;
 *  - ticket-tagged answers: the mailbox and the ring's `done` word carry the full 32-bit tag (ticket + 1), and a
 *    waiter accepts only its own tag, so a stale or aliased leader write is ignored;
 *  - bounded waits: a claimed waiter sleeps in QSB_RC_SLEEP_NS chunks until QSB_RC_SERVE_CYC SM cycles after it
 *    saw the claim (the leader's service is about 17 to 19 us at 1.71 GHz; __nanosleep alone only promises
 *    [0, 2t]), then polls every QSB_RC_DONE_NS, and after QSB_RC_WAIT_CYC it inverts its own root. A protocol
 *    fault can cost time or (for a late write racing a 4096-ticket alias) that block's hits, never a hang and
 *    never a published wrong hit (the host gate re-derives every hit);
 *  - arch guard and knob entries: included only into the native sm_89 image (tree_inverse.cuh), every QSB_RC_*
 *    default at file scope in tree.cu and in QSB_CARRIER_KNOBS.
 * Not done: QSB_RC_STATS / QSB_RC_CHECK counters and their host readout, the __noinline__ isolation attempt, the
 * (blockIdx, stream parity) ring slot, the operand-order search on the final stack.
 *
 * Roles (lane 0 decides, the warp follows by vote):
 *  - poster: tag = ticket + 1, root to ring slot (tag - 1) & (RING - 1), done = 0, release-CAS mbox 0 -> tag;
 *  - leader: finds mbox = other tag, acquire-CAS it back to 0, loads the partner root, inverts R_self * R_partner
 *    once, keeps inv * R_partner (its own inverse) and releases inv * R_self to the partner with done = tag;
 *  - waiter (a poster that was claimed): waits for done == its tag, then reads its inverse;
 *  - solo: nobody to pair with, a withdrawn poster, a timed-out waiter or ticket 0xffffffff (tag 0 is reserved). */
struct __align__(16) qsb_rc_rec_t { unsigned long long v[4]; unsigned long long inv[4]; unsigned done; unsigned pad[3]; };
__device__ qsb_rc_rec_t qsb_rc_ring[QSB_RC_RING];
__device__ unsigned qsb_rc_ticket;
__device__ unsigned qsb_rc_mbox;
__device__ __forceinline__ unsigned qsb_rc_ld_rlx(const unsigned *p){
    unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void qsb_rc_st_rel(unsigned *p,unsigned v){
    asm volatile("st.release.gpu.global.u32 [%0],%1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned qsb_rc_cas_rel(unsigned *p,unsigned c,unsigned v){
    unsigned o; asm volatile("atom.release.gpu.global.cas.b32 %0,[%1],%2,%3;" : "=r"(o) : "l"(p),"r"(c),"r"(v) : "memory"); return o; }
__device__ __forceinline__ unsigned qsb_rc_cas_acq(unsigned *p,unsigned c,unsigned v){
    unsigned o; asm volatile("atom.acquire.gpu.global.cas.b32 %0,[%1],%2,%3;" : "=r"(o) : "l"(p),"r"(c),"r"(v) : "memory"); return o; }
__device__ __forceinline__ void qsb_rc_fence_acq(){ asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void qsb_rc_mul_exact(uint64_t *o,uint64_t *a,uint64_t *b){
    qsb_field_mul_raw(o,a,b); qsb_field_normalize(o); o[4]=0; }
/* root[0..3] holds the block's canonical root on every lane of warp 0 (root[4] = 0). On return lane 0 holds
 * inv(u)/root, the value zi_inverse_limbs(root, lt) leaves on lane 0; the caller broadcasts lane 0's value. */
__device__ __forceinline__ void qsb_root_combine_invert(uint64_t *root,int lt){
    constexpr unsigned F=0xffffffffu;
    unsigned info=0,tag=0;         /* lane 0: 0x40000000 posted, 0x80000000|partner tag in `ptag` leader, 0 solo */
    unsigned ptag=0;
    if(lt==0){
        tag=atomicAdd(&qsb_rc_ticket,1u)+1u;                     /* 0 only for ticket 0xffffffff: stay solo */
        if(tag!=0u){
            qsb_rc_rec_t *r=&qsb_rc_ring[(tag-1u)&(QSB_RC_RING-1)];
            #pragma unroll
            for(int k=0;k<4;k++)__stcg(&r->v[k],(unsigned long long)root[k]);
            __stcg(&r->done,0u);
            const unsigned old=qsb_rc_cas_rel(&qsb_rc_mbox,0u,tag);
            if(old==0u)info=0x40000000u;
            else if(qsb_rc_cas_acq(&qsb_rc_mbox,old,0u)==old){info=0x80000000u;ptag=old;}
        }
    }
    const bool posted=__any_sync(F,(info&0x40000000u)!=0u);
    if(posted){
        bool claimed=false;
        #pragma unroll 1
        for(int i=0;i<QSB_RC_CLAIM_POLLS;i++){
            __nanosleep(QSB_RC_CLAIM_NS);
            unsigned c=0; if(lt==0)c=(qsb_rc_ld_rlx(&qsb_rc_mbox)!=tag);
            if(__any_sync(F,c!=0u)){claimed=true;break;}
        }
        if(!claimed){
            unsigned w=0; if(lt==0)w=(atomicCAS(&qsb_rc_mbox,tag,0u)!=tag);   /* 1 = withdraw failed: claimed */
            claimed=__any_sync(F,w!=0u);
        }
        if(claimed){
            long long tc=0; if(lt==0)tc=clock64();
            #pragma unroll 1
            for(;;){                                   /* sleep through the leader's service */
                unsigned more=0; if(lt==0)more=(clock64()-tc<(long long)QSB_RC_SERVE_CYC);
                if(!__any_sync(F,more!=0u))break;
                __nanosleep(QSB_RC_SLEEP_NS);
            }
            bool served=false;
            #pragma unroll 1
            for(;;){                                   /* then poll for this block's own tag, bounded */
                unsigned d=0,tmo=0;
                if(lt==0){ d=(qsb_rc_ld_rlx(&qsb_rc_ring[(tag-1u)&(QSB_RC_RING-1)].done)==tag);
                           tmo=(clock64()-tc>=(long long)QSB_RC_WAIT_CYC); }
                if(__any_sync(F,d!=0u)){served=true;break;}
                if(__any_sync(F,tmo!=0u))break;
                __nanosleep(QSB_RC_DONE_NS);
            }
            if(served){
                if(lt==0){
                    qsb_rc_fence_acq();
                    const qsb_rc_rec_t *r=&qsb_rc_ring[(tag-1u)&(QSB_RC_RING-1)];
                    #pragma unroll
                    for(int k=0;k<4;k++)root[k]=__ldcg(&r->inv[k]);
                }
                root[4]=0;
                return;                                /* the caller broadcasts lane 0's value */
            }
            /* timed out: fall through to the solo inversion of the block's own root */
        }
    }
    const bool lead=__any_sync(F,(info&0x80000000u)!=0u);
    uint64_t x[5],pv[5];
    #pragma unroll
    for(int k=0;k<5;k++){x[k]=root[k];pv[k]=0;}
    if(lead){
        if(lt==0){
            const qsb_rc_rec_t *r=&qsb_rc_ring[(ptag-1u)&(QSB_RC_RING-1)];
            #pragma unroll
            for(int k=0;k<4;k++)pv[k]=__ldcg(&r->v[k]);
        }
        #pragma unroll
        for(int k=0;k<4;k++)pv[k]=__shfl_sync(F,pv[k],0);
        qsb_rc_mul_exact(x,root,pv);               /* R_self * R_partner, canonical */
    }
    zi_inverse_limbs(x,lt);                        /* single call site: inv(u)/x on every lane */
    if(lead){
        uint64_t o[5];
        #pragma unroll
        for(int k=0;k<5;k++)o[k]=(lt==1)?root[k]:pv[k];   /* lane 0: own inverse; lane 1: the partner's */
        qsb_rc_mul_exact(o,x,o);
        const unsigned pt=__shfl_sync(F,ptag,0);
        if(lt==1){
            qsb_rc_rec_t *r=&qsb_rc_ring[(pt-1u)&(QSB_RC_RING-1)];
            #pragma unroll
            for(int k=0;k<4;k++)__stcg(&r->inv[k],(unsigned long long)o[k]);
            qsb_rc_st_rel(&r->done,pt);
        }
        #pragma unroll
        for(int k=0;k<4;k++)x[k]=o[k];
    }
    #pragma unroll
    for(int k=0;k<4;k++)root[k]=x[k];
    root[4]=0;
}
