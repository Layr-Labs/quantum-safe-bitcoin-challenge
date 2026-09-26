/* Optional host-only 13-window signed recovery. No GPU/carrier changes.
 * Signed and folded-table direction: promoted Subset a137e28. Pinning adapter
 * is independent. Exceptional points use complete OpenSSL recovery, never
 * omission. Startup compares complete output coordinates before timing.
 */
#pragma once
namespace sf {
using namespace vi;
namespace plan = qcg_signed_plan;
#define QFT __attribute__((target("avx512f,avx512ifma")))
struct state { uint32_t d[QSB_CG_MAXB][plan::windows]; };

/* A whole batch (or its remaining recid) retries before it has published any
 * outputs for that recid. This also handles z=0, signed cancellation, doubling,
 * folded infinity and a zero chain root without contaminating other lanes. */
static bool complete(worker_t *w,int first_recid,tentry *capture) {
    bool ok=true;
    BN_CTX_start(w->ctx);
    BIGNUM *z=BN_CTX_get(w->ctx),*k=BN_CTX_get(w->ctx);
    BIGNUM *x=BN_CTX_get(w->ctx),*y=BN_CTX_get(w->ctx);
    EC_POINT *p=EC_POINT_new(w->grp),*q=EC_POINT_new(w->grp);
    if(!y || !p || !q) ok=false;
    for(int i=0;ok && i<w->n;++i) {
        ok=BN_lebin2bn((const unsigned char *)w->z[i],32,z) &&
           BN_mod_mul(k,z,w->nri,w->order,w->ctx) &&
           EC_POINT_mul(w->grp,p,k,NULL,NULL,w->ctx);
        for(int r=first_recid;ok && r<2;++r) {
            ok=EC_POINT_copy(q,w->Ru2);
            if(ok && r)ok=EC_POINT_invert(w->grp,q,w->ctx);
            if(ok)ok=EC_POINT_add(w->grp,q,p,q,w->ctx);
            if(!ok)break;
            const int infinity=EC_POINT_is_at_infinity(w->grp,q);
            if(capture) {
                tentry &e=capture[r*QSB_CG_MAXB+i];
                memset(&e,0,sizeof e);
                if(!infinity) {
                    ok=EC_POINT_get_affine_coordinates(w->grp,q,x,y,w->ctx) &&
                       BN_bn2lebinpad(x,(unsigned char *)e.x,32)==32 &&
                       BN_bn2lebinpad(y,(unsigned char *)e.y,32)==32;
                }
            } else if(!infinity) {
                uint8_t block[64]={}; uint32_t h[8];
                ok=EC_POINT_point2oct(w->grp,q,POINT_CONVERSION_COMPRESSED,block,33,w->ctx)==33;
                if(!ok)break;
                block[33]=0x80;block[62]=1;block[63]=8;
                memcpy(h,SHA_IV,32);sha_blocks(h,block,1);
                if(lz_ok(h))publish(w,i,r);
            }
        }
    }
    EC_POINT_free(p);EC_POINT_free(q);BN_CTX_end(w->ctx);
    if(!ok)g_cg->failed.store(1);
    return ok;
}

QFT static void emit(worker_t *w,int b,int r,const fe8 &x,const fe8 &y,tentry *capture) {
    if(!capture) { vi::hash_keys(w,b,r,x,y);return; }
    __m512i a[4],c[4];fe8_canon_words(a,x);fe8_canon_words(c,y);
    alignas(64) uint64_t ax[4][8],ay[4][8];
    for(int k=0;k<4;++k) {_mm512_store_si512(ax[k],a[k]);_mm512_store_si512(ay[k],c[k]);}
    for(int l=0;l<8 && b*8+l<w->n;++l) {
        tentry &e=capture[r*QSB_CG_MAXB+b*8+l];
        for(int k=0;k<4;++k) {e.x[k]=ax[k][l];e.y[k]=ay[k][l];}
    }
}
static inline bool infinity_row(const tentry *e) {
    return !(e->x[0]|e->x[1]|e->x[2]|e->x[3]|e->y[0]|e->y[1]|e->y[2]|e->y[3]);
}
QFT static inline void prefetch(worker_t *w,state *f,int b,int j) {
    for(int l=0;l<8 && b*8+l<w->n;++l) {
        const int i=b*8+l;
        if(j<plan::windows-1)
            __builtin_prefetch(g_cg->signed_table+plan::offset(j)+plan::magnitude(f->d[i][j]),0,1);
        else for(int r=0;r<2;++r)
            __builtin_prefetch(g_cg->signed_table+plan::folded_index(f->d[i][12],r),0,1);
    }
}
QFT static bool ec_batch(worker_t *w,vi::vstate *s,state *f,tentry *capture=NULL) {
    const int n=w->n,nb=(n+7)/8;
    const tentry *T=g_cg->signed_table;
    fe8 acc[2],inv[2],identity;one(identity);
    for(int i=0;i<n;++i)plan::digits(f->d[i],w->z[i]);
    for(int b=0;b<nb;++b) {
        const tentry *row[8];__mmask8 sign=0;
        for(int l=0;l<8;++l) {
            const int i=b*8+l;const uint32_t d=i<n?f->d[i][0]:0;
            row[l]=T+plan::magnitude(d);w->inf[i]=!plan::magnitude(d);
            if(plan::negative(d))sign|=(__mmask8)(1u<<l);
        }
        pt8_load(s->x[b],s->y[b],row);fe8_cneg(s->y[b],sign);
        prefetch(w,f,b,1);
    }
    for(int j=1;j<plan::windows-1;++j) {
        one(acc[0]);one(acc[1]);const tentry *Tj=T+plan::offset(j);
        for(int b=0;b<nb;++b) {
            const tentry *row[8];__mmask8 am=0,lm=0,sign=0;
            for(int l=0;l<8;++l) {
                const int i=b*8+l;const uint32_t d=i<n?f->d[i][j]:0;
                const uint32_t mag=plan::magnitude(d);row[l]=Tj+mag;
                if(plan::negative(d))sign|=(__mmask8)(1u<<l);
                if(i<n && mag) {
                    if(w->inf[i])lm|=(__mmask8)(1u<<l);else am|=(__mmask8)(1u<<l);
                    w->inf[i]=0;
                }
            }
            fe8 tx,ty,d;pt8_load(tx,ty,row);
            fe8_sub(d,tx,s->x[b]);select(s->dx[b],am,d,identity);
            fe8_sub_sgn(s->ty[b],ty,s->y[b],sign);
            if(lm) {fe8_cneg(ty,sign);select(s->x[b],lm,tx,s->x[b]);select(s->y[b],lm,ty,s->y[b]);}
            s->act[b]=am;fe8_mul(acc[b&1],acc[b&1],s->dx[b]);fe8_mov(s->c[b],acc[b&1]);
        }
        if(vi::chain_invert(inv,acc))return complete(w,0,capture);
        for(int b=nb-1;b>=0;--b) {
            const int g=b&1;fe8 ik,lam,x3,y3,t;
            prefetch(w,f,nb-1-b,j+1);
            if(b>=2)fe8_mul(ik,inv[g],s->c[b-2]);else fe8_mov(ik,inv[g]);
            fe8_mul(inv[g],inv[g],s->dx[b]);fe8_mul(lam,s->ty[b],ik);
            fe8_sqr_sub3(x3,lam,s->dx[b],s->x[b]);
            fe8_sub_m(t,s->x[b],x3);fe8_mul_sub(y3,lam,t,s->y[b]);
            select(s->x[b],s->act[b],x3,s->x[b]);select(s->y[b],s->act[b],y3,s->y[b]);
        }
    }
    /* Sequential recids reuse the compact state. One extra scalar root versus
     * concatenating 2*n denominators saves three expanded field-state arrays.
     * A retry for r=1 never republishes r=0, which has already completed. */
    for(int r=0;r<2;++r) {
        one(acc[0]);one(acc[1]);
        for(int b=0;b<nb;++b) {
            const tentry *row[8];__mmask8 active=0;
            for(int l=0;l<8;++l) {
                const int i=b*8+l;const uint32_t d=i<n?f->d[i][12]:0;
                row[l]=T+plan::folded_index(d,r);
                if(i<n) {
                    if(w->inf[i] || (g_cg->signed_infinity[r] && infinity_row(row[l])))return complete(w,r,capture);
                    active|=(__mmask8)(1u<<l);
                }
            }
            fe8 tx,ty,d;pt8_load(tx,ty,row);
            fe8_sub(d,tx,s->x[b]);select(s->dx[b],active,d,identity);
            fe8_sub_m(s->ty[b],ty,s->y[b]);
            fe8_mul(acc[b&1],acc[b&1],s->dx[b]);fe8_mov(s->c[b],acc[b&1]);
        }
        if(vi::chain_invert(inv,acc))return complete(w,r,capture);
        for(int b=nb-1;b>=0;--b) {
            const int g=b&1;fe8 ik,lam,qx,qy,t;
            if(b>=2)fe8_mul(ik,inv[g],s->c[b-2]);else fe8_mov(ik,inv[g]);
            fe8_mul(inv[g],inv[g],s->dx[b]);fe8_mul(lam,s->ty[b],ik);
            fe8_sqr_sub3(qx,lam,s->dx[b],s->x[b]);
            fe8_sub_m(t,s->x[b],qx);fe8_mul_sub(qy,lam,t,s->y[b]);emit(w,b,r,qx,qy,capture);
        }
    }
    return true;
}
QFT static bool selfcheck(worker_t *w,vi::vstate *s,state *f) {
    tentry *want=(tentry *)calloc(2*QSB_CG_MAXB,sizeof(tentry));
    tentry *got=(tentry *)calloc(2*QSB_CG_MAXB,sizeof(tentry));
    if(!want || !got) {free(want);free(got);return false;}
    uint64_t rng=0x7369676e6564666fULL;bool ok=true;
    w->n=41;
    for(int i=0;i<w->n;++i)for(int k=0;k<4;++k) {
        rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;w->z[i][k]=rng;
    }
    /* Dense case is first, so complete fallback cannot conceal the fast path. */
    ok=complete(w,0,want) && ec_batch(w,s,f,got);
    for(int r=0;ok && r<2;++r)ok=!memcmp(want+r*QSB_CG_MAXB,got+r*QSB_CG_MAXB,w->n*sizeof(tentry));
    for(int pass=0;ok && pass<3;++pass) {
        memset(w->z,0,sizeof w->z);w->n=pass==0?1:pass==1?17:33;
        for(int i=0;i<w->n;++i) {
            if(pass==1)w->z[i][(i*17)&3]=1ULL<<((i*19)&63);
            if(pass==2)for(int k=0;k<4;++k)w->z[i][k]=~0ULL;
        }
        ok=complete(w,0,want) && ec_batch(w,s,f,got);
        for(int r=0;ok && r<2;++r)ok=!memcmp(want+r*QSB_CG_MAXB,got+r*QSB_CG_MAXB,w->n*sizeof(tentry));
    }
    free(want);free(got);w->n=0;return ok;
}
#undef QFT
} // namespace sf
