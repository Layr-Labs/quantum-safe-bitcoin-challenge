/* Pinning host IFMA pipeline, adapted independently to the promoted co-grinder.
 * Field primitives: promoted Subset a137e28; API/enumeration/gate: Pinning cc75e3b.
 * Full zero digits and inactive lanes retain the base's point-at-infinity state.
 * One table read per window, fused ordinate updates, backwards next-window prefetch.
 */
#pragma once
namespace vi {
#include "cpu_ifma_field.h"
struct vstate {
    fe8 x[QSB_CG_MAXB/8], y[QSB_CG_MAXB/8], c[QSB_CG_MAXB/8];
    fe8 dx[QSB_CG_MAXB/8], ty[QSB_CG_MAXB/8];
    __mmask8 act[QSB_CG_MAXB/8];
};
Q8T static inline void one(fe8 &r) {
    r.l[0] = _mm512_set1_epi64(1);
    for (int k = 1; k < 5; ++k) r.l[k] = _mm512_setzero_si512();
}
Q8T static inline void select(fe8 &r, __mmask8 m, const fe8 &a, const fe8 &b) {
    for (int k = 0; k < 5; ++k) r.l[k] = _mm512_mask_mov_epi64(b.l[k], m, a.l[k]);
}
Q8T static inline void broadcast(fe8 &r, const fe &x) {
    fe c=x; fe_normalize(&c);
    for (int k=0;k<5;++k) r.l[k]=_mm512_set1_epi64(c.n[k]);
}
Q8T static inline void swap_lanes(fe8 &r, const fe8 &a, int stride) {
    const __m512i ix = _mm512_xor_si512(_mm512_set_epi64(7,6,5,4,3,2,1,0),_mm512_set1_epi64(stride));
    for (int k=0;k<5;++k) r.l[k]=_mm512_permutexvar_epi64(ix,a.l[k]);
}
Q8T static void chain_invert(fe8 out[2], const fe8 acc[2]) {
    fe8 m,q,other,s,t,identity; one(identity);
    fe8_mul(m,acc[0],acc[1]);
    __m512i words[4]; fe8_canon_words(words,m);
    const __m512i nz=_mm512_or_si512(_mm512_or_si512(words[0],words[1]),_mm512_or_si512(words[2],words[3]));
    const __mmask8 zero=_mm512_cmpeq_epi64_mask(nz,_mm512_setzero_si512());
    select(q,zero,identity,m);
    for(int stride=1;stride<8;stride<<=1) {
        swap_lanes(s,q,stride);
        if(stride==1) fe8_mov(other,s); else fe8_mul(other,other,s);
        fe8_mul(q,q,s);
    }
    fe8_canon_words(words,q);
    uint64_t w[4];
    for(int k=0;k<4;++k) w[k]=(uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(words[k]));
    fe a,inv; fe_from_w(&a,w); fe_inv(&inv,&a); broadcast(t,inv);
    fe8_mul(t,t,other);
    fe8_mul(out[0],t,acc[1]); fe8_mul(out[1],t,acc[0]);
    for(int k=0;k<5;++k) {
        out[0].l[k]=_mm512_mask_mov_epi64(out[0].l[k],zero,_mm512_setzero_si512());
        out[1].l[k]=_mm512_mask_mov_epi64(out[1].l[k],zero,_mm512_setzero_si512());
    }
}
Q8T static inline void prefetch_group(worker_t *w,int group,int window) {
    if(window>=QSB_CG_NWIN) return;
    const tentry *T=g_cg->table+(size_t)window*QSB_CG_TSIZE;
    for(int l=0;l<8;++l) {
        int i=group*8+l;
        if(i<w->n) __builtin_prefetch(T+digit(w->z[i],window),0,1);
    }
}
Q8T static void hash_keys(worker_t *w,int group,int recid,const fe8 &x,const fe8 &y) {
    __m512i xw[4],yw[4]; fe8_canon_words(xw,x); fe8_canon_words(yw,y);
    alignas(64) uint64_t wx[4][8],yp[8];
    for(int k=0;k<4;++k) _mm512_store_si512(wx[k],xw[k]);
    _mm512_store_si512(yp,yw[0]);
    for(int l0=0;l0<8;l0+=4) {
        uint8_t blk[4][64] = {}; const uint8_t *ptr[4]; uint32_t st[4][8];
        for(int l=0;l<4;++l) {
            const int ln=l0+l;
            blk[l][0]=(uint8_t)(2|(yp[ln]&1));
            for(int k=0;k<4;++k) for(int b=0;b<8;++b)
                blk[l][1+8*k+b]=(uint8_t)(wx[3-k][ln]>>(56-8*b));
            blk[l][33]=0x80; blk[l][62]=1; blk[l][63]=8;
            memcpy(st[l],SHA_IV,32); ptr[l]=blk[l];
        }
        cpu_hash4(st,ptr);
        for(int l=0;l<4;++l) {
            int i=group*8+l0+l;
            if(i<w->n && !w->inf[i] && lz_ok(st[l])) publish(w,i,recid);
        }
    }
}
Q8T static int field_selfcheck() {
    uint64_t rng=0x9e3779b97f4a7c15ULL;
    for(int round=0;round<24;++round) {
        alignas(64) tentry rows[8];const tentry *ptr[8];
        for(int l=0;l<8;++l) {
            ptr[l]=rows+l;
            for(int k=0;k<4;++k) {
                rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;rows[l].x[k]=rng;
                rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;rows[l].y[k]=rng;
            }
            if(round<8 && l==round) memset(rows[l].x,0,32);
        }
        fe8 a,b,got,d,acc[2],inv[2];pt8_load(a,b,ptr);
        fe8_mul(got,a,b);
        __m512i gw[4];fe8_canon_words(gw,got);alignas(64) uint64_t out[4][8];
        for(int k=0;k<4;++k)_mm512_store_si512(out[k],gw[k]);
        for(int l=0;l<8;++l) {
            fe sa,sb,sr;uint64_t want[4];fe_from_w(&sa,rows[l].x);fe_from_w(&sb,rows[l].y);
            fe_mul(&sr,&sa,&sb);fe_normalize(&sr);fe_to_w(want,&sr);
            for(int k=0;k<4;++k)if(out[k][l]!=want[k])return 0;
        }
        fe8_mov(acc[0],a);fe8_mov(acc[1],b);chain_invert(inv,acc);
        fe8_canon_words(gw,inv[0]);for(int k=0;k<4;++k)_mm512_store_si512(out[k],gw[k]);
        for(int l=0;l<8;++l) {
            fe sa,sb,sr;uint64_t want[4];fe_from_w(&sa,rows[l].x);fe_from_w(&sb,rows[l].y);
            fe_normalize(&sa);fe_normalize(&sb);
            if(fe_is_zero_norm(&sa)||fe_is_zero_norm(&sb))memset(want,0,sizeof want);
            else {fe_inv_fermat(&sr,&sa);fe_normalize(&sr);fe_to_w(want,&sr);}
            for(int k=0;k<4;++k)if(out[k][l]!=want[k])return 0;
        }
    }
    return 1;
}
Q8T static void ec_batch(worker_t *w,vstate *s) {
    shared_t *S=g_cg; const int n=w->n,nb=(n+7)/8; const tentry *T=S->table;
    fe8 acc[2],inv[2],identity; one(identity);
    for(int b=0;b<nb;++b) {
        const tentry *row[8];
        for(int l=0;l<8;++l) {
            int i=b*8+l; unsigned d=i<n?digit(w->z[i],0):0;
            row[l]=T+d; w->inf[i]=!d;
        }
        pt8_load(s->x[b],s->y[b],row); prefetch_group(w,b,1);
    }
    for(int j=1;j<QSB_CG_NWIN;++j) {
        one(acc[0]);one(acc[1]); const tentry *Tj=T+(size_t)j*QSB_CG_TSIZE;
        for(int b=0;b<nb;++b) {
            const tentry *row[8]; __mmask8 am=0,lm=0;
            for(int l=0;l<8;++l) {
                const int i=b*8+l; unsigned d=i<n?digit(w->z[i],j):0;
                row[l]=Tj+d;
                if(i<n && d) {
                    if(w->inf[i]) lm|=(__mmask8)(1u<<l); else am|=(__mmask8)(1u<<l);
                    w->inf[i]=0;
                }
            }
            fe8 tx,ty,d; pt8_load(tx,ty,row);
            fe8_sub(d,tx,s->x[b]); select(s->dx[b],am,d,identity);
            fe8_sub_m(s->ty[b],ty,s->y[b]);
            /* Point-at-infinity loads happen before the reverse pass. Its active mask is false. */
            select(s->x[b],lm,tx,s->x[b]);select(s->y[b],lm,ty,s->y[b]);
            s->act[b]=am;
            fe8_mul(acc[b&1],acc[b&1],s->dx[b]); fe8_mov(s->c[b],acc[b&1]);
        }
        chain_invert(inv,acc);
        for(int b=nb-1;b>=0;--b) {
            const int g=b&1; fe8 ik,lam,x3,y3,t;
            /* Reverse visitation prefetches groups 0,1,... in next window consumption order. */
            prefetch_group(w,nb-1-b,j+1);
            if(b>=2) fe8_mul(ik,inv[g],s->c[b-2]);else fe8_mov(ik,inv[g]);
            fe8_mul(inv[g],inv[g],s->dx[b]);
            fe8_mul(lam,s->ty[b],ik);
            fe8_sqr_sub3(x3,lam,s->dx[b],s->x[b]);
            fe8_sub_m(t,s->x[b],x3);fe8_mul_sub(y3,lam,t,s->y[b]);
            select(s->x[b],s->act[b],x3,s->x[b]);select(s->y[b],s->act[b],y3,s->y[b]);
        }
    }
    fe8 ax,ay;broadcast(ax,S->ax);broadcast(ay,S->ay);
    one(acc[0]);one(acc[1]);
    for(int b=0;b<nb;++b) {
        __mmask8 am=0;
        for(int l=0;l<8;++l) if(b*8+l<n && !w->inf[b*8+l]) am|=(__mmask8)(1u<<l);
        fe8 d;fe8_sub(d,ax,s->x[b]);select(s->dx[b],am,d,identity);
        fe8_mul(acc[b&1],acc[b&1],s->dx[b]);fe8_mov(s->c[b],acc[b&1]);
    }
    chain_invert(inv,acc);
    for(int b=nb-1;b>=0;--b) {
        const int g=b&1;fe8 ik;
        if(b>=2)fe8_mul(ik,inv[g],s->c[b-2]);else fe8_mov(ik,inv[g]);
        fe8_mul(inv[g],inv[g],s->dx[b]);
        for(int recid=0;recid<2;++recid) {
            fe8 ty,lam,qx,qy,t;
            fe8_sub_sgn(ty,ay,s->y[b],recid?0xff:0);
            fe8_mul(lam,ty,ik);fe8_sqr_sub3(qx,lam,s->dx[b],s->x[b]);
            fe8_sub_m(t,s->x[b],qx);fe8_mul_sub(qy,lam,t,s->y[b]);
            hash_keys(w,b,recid,qx,qy);
        }
    }
}
#undef Q8T
#undef F8_M52
} // namespace vi
