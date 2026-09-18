/* CPU cross-check of the table-driven Bernstein-Yang root inverse, compiled
 * from the SHIPPED source text of tests/gpu_epochs/zinv32.cuh (lane_by.cpp and
 * lane_st.cpp #include that file verbatim through its own !__CUDA_ARCH__
 * branch; nothing is retyped).  Four real OS threads run lanes 0..3 in lockstep
 * at every warp primitive -- see warp.h for exactly what is and is not modelled.
 * References: OpenSSL BN_mod_inverse, and the shipped ZLAB_BY=0 Stein path.
 * Gate: x * inv(x) == 1 (mod p), zero failures. */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <pthread.h>
#include <openssl/bn.h>
#include "warp.h"

/* ---------------------------------------------- 4-lane lockstep warp -------- */
#define QSB_LANES 4
static std::atomic<unsigned> g_bar_count(0), g_bar_gen(0);
static void warp_barrier(void){
    unsigned gen = g_bar_gen.load(std::memory_order_acquire);
    if (g_bar_count.fetch_add(1, std::memory_order_acq_rel) == QSB_LANES-1) {
        g_bar_count.store(0, std::memory_order_relaxed);
        g_bar_gen.store(gen+1, std::memory_order_release);
    } else {
        while (g_bar_gen.load(std::memory_order_acquire) == gen) __builtin_ia32_pause();
    }
}
static volatile uint32_t g_slotv[QSB_LANES];
static volatile int      g_slots[QSB_LANES];
static thread_local int  t_lane = 0;
static long long g_shfl = 0;          /* shuffles in the current inversion */
static uint32_t  g_fneg = 0;          /* value broadcast by the uniform-src-0 shuffle */
static long long g_shfl_limit = 12*64;
uint32_t qsb_warp_shfl(uint32_t v, int src){
    if (src < 0 || src >= QSB_LANES) { fprintf(stderr,"FATAL: srcLane %d outside the 0xF mask\n",src); abort(); }
    g_slotv[t_lane] = v; g_slots[t_lane] = src;
    warp_barrier();
    uint32_t out = g_slotv[src];
    if (t_lane == 0) {
        if (++g_shfl > g_shfl_limit) { fprintf(stderr,"FATAL: inverse did not terminate (%lld shuffles)\n",g_shfl); abort(); }
        if (g_slots[0]==0 && g_slots[1]==0 && g_slots[2]==0 && g_slots[3]==0) g_fneg = g_slotv[0];
    }
    warp_barrier();
    return out;
}
void by_inverse(uint64_t*,int);
void st_inverse(uint64_t*,int);
int32_t by_divstep30(int32_t,uint32_t,uint32_t,int32_t*,int32_t*,int32_t*,int32_t*);

/* ------------------------------------ independent divstep reference --------- */
/* Direct Bernstein-Yang recursion in int64, no table, no clamp: the reference
 * the shipped 7-lookup + 2-single-step routine must reproduce exactly.  The
 * 30-step case sequence depends only on (delta, f mod 2^32, g mod 2^32). */
static int32_t ref_divstep30(int32_t delta, uint32_t f0, uint32_t g0,
                             int64_t *ru,int64_t *rv,int64_t *rq,int64_t *rr){
    int64_t f=(int64_t)(uint64_t)f0, g=(int64_t)(uint64_t)g0;
    int64_t u=1,v=0,q=0,r=1;
    for(int i=0;i<30;i++){
        if(delta>0 && (g&1)){
            int64_t nf=g, ng=(g-f)/2, nu=2*q, nv=2*r, nq=q-u, nr=r-v;
            delta=1-delta; f=nf; g=ng; u=nu; v=nv; q=nq; r=nr;
        } else if(g&1){
            int64_t ng=(g+f)/2, nu=2*u, nv=2*v, nq=q+u, nr=r+v;
            delta=1+delta; g=ng; u=nu; v=nv; q=nq; r=nr;
        } else {
            delta=1+delta; g=g/2; u=2*u; v=2*v;
        }
    }
    *ru=u;*rv=v;*rq=q;*rr=r; return delta;
}

/* ------------------------------------------------------- test vectors ------- */
static const uint64_t PL[4]={0xFFFFFFFEFFFFFC2FULL,~0ULL,~0ULL,~0ULL};
static uint64_t splitmix(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static int lt_p(const uint64_t*x){ for(int i=3;i>=0;i--) if(x[i]!=PL[i]) return x[i]<PL[i]; return 0; }
static void sub_p(uint64_t*x){ __uint128_t b=0; for(int i=0;i<4;i++){ __uint128_t t=(__uint128_t)x[i]-PL[i]-b; x[i]=(uint64_t)t; b=(t>>64)&1; } }
static void red_p(uint64_t*x){ while(!lt_p(x)) sub_p(x); }
static void sub_small(uint64_t*x,uint64_t k){ __uint128_t b=k; for(int i=0;i<4&&b;i++){ __uint128_t t=(__uint128_t)x[i]-(uint64_t)b; x[i]=(uint64_t)t; b=(t>>64)&1; } }

static long long N_RAND = 200000;
enum { C_RAND, C_SMALL, C_PM, C_POW, C_LIMBPAT, C_ZERO, C_EVEN2J, C_HIB, C_NCLS };
static const char* cls_name[C_NCLS]={
 "rand in [1,p)","small 1..4096","p-1 .. p-4096","2^k-1,2^k,2^k+1 (k=0..255)",
 "limb patterns (all-ones/zero/AA/55)","zero (non-invertible)",
 "odd*2^j, j=0..255  (long even runs -> delta far above the clamp)",
 "max-batch search (19-batch witnesses + random)"};
static long long cls_lo[C_NCLS], cls_hi[C_NCLS];
static void build_ranges(void){
    long long o=0;
    cls_lo[C_RAND]=o;    o+=N_RAND;      cls_hi[C_RAND]=o;
    cls_lo[C_SMALL]=o;   o+=4096;        cls_hi[C_SMALL]=o;
    cls_lo[C_PM]=o;      o+=4096;        cls_hi[C_PM]=o;
    cls_lo[C_POW]=o;     o+=3*256;       cls_hi[C_POW]=o;
    cls_lo[C_LIMBPAT]=o; o+=4*256;       cls_hi[C_LIMBPAT]=o;
    cls_lo[C_ZERO]=o;    o+=1;           cls_hi[C_ZERO]=o;
    cls_lo[C_EVEN2J]=o;  o+=256*16;      cls_hi[C_EVEN2J]=o;
    cls_lo[C_HIB]=o;     o+=N_RAND/2;    cls_hi[C_HIB]=o;
}
static int which_class(long long i){ for(int c=0;c<C_NCLS;c++) if(i>=cls_lo[c]&&i<cls_hi[c]) return c; return -1; }
/* pure function of the case index: every lane derives the same input */
static void make_input(long long i, uint64_t *x){
    x[0]=x[1]=x[2]=x[3]=0;
    int c=which_class(i); long long k=i-cls_lo[c];
    uint64_t s;
    switch(c){
    case C_RAND:   s=0x5EEDULL*1000003ULL+(uint64_t)k;
                   for(int j=0;j<4;j++) x[j]=splitmix(&s);
                   red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1; break;
    case C_SMALL:  x[0]=(uint64_t)k+1; break;
    case C_PM:     memcpy(x,PL,32); sub_small(x,(uint64_t)k+1); break;
    case C_POW:  { int kk=(int)(k/3), w=(int)(k%3);
                   uint64_t t[4]={0,0,0,0}; t[kk>>6]=1ULL<<(kk&63);
                   if(w==0){ /* 2^kk - 1 */ memset(t,0,32);
                       for(int b=0;b<kk;b++) t[b>>6]|=1ULL<<(b&63); }
                   else if(w==2){ __uint128_t cy=1; for(int j=0;j<4&&cy;j++){ __uint128_t q=(__uint128_t)t[j]+(uint64_t)cy; t[j]=(uint64_t)q; cy=q>>64; } }
                   memcpy(x,t,32); red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1; break; }
    case C_LIMBPAT:{ static const uint64_t pat[4]={0ULL,~0ULL,0xAAAAAAAAAAAAAAAAULL,0x5555555555555555ULL};
                   int m=(int)(k&255), pi=(int)(k>>8);
                   for(int j=0;j<4;j++) x[j]=((m>>j)&1)?pat[pi]:0ULL;
                   red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1; break; }
    case C_ZERO:   break;
    case C_EVEN2J:{ int j=(int)(k>>4); s=0xABCDEFULL+(uint64_t)k;
                   uint64_t o[4]; for(int t2=0;t2<4;t2++) o[t2]=splitmix(&s);
                   o[0]|=1ULL;                       /* odd multiplier */
                   /* x = (o mod 2^(256-j)) << j  : j trailing zero bits */
                   for(int b=255;b>=0;b--){ int sb=b-j; if(sb<0){ x[b>>6]&=~(1ULL<<(b&63)); continue; }
                       if((o[sb>>6]>>(sb&63))&1ULL) x[b>>6]|=1ULL<<(b&63); else x[b>>6]&=~(1ULL<<(b&63)); }
                   red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1; break; }
    default:       s=0xD1CEULL*7919ULL+(uint64_t)k*31ULL;
                   for(int j=0;j<4;j++) x[j]=splitmix(&s);
                   red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1; break;
    }
}

/* ------------------------------------------------------------ harness ------- */
static long long N_CASES;
static uint64_t g_by[5], g_st[5];
static std::atomic<long long> g_fail(0);
static long long g_batch_hist[64];
static long long g_fneg_cnt, g_checked, g_zero_out;
static long long g_maxbatch;
static long long g_stein_hist[64];
static const char *PHEX="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";

static void bn_from_limbs(BIGNUM*b,const uint64_t*x){ uint8_t be[32];
    for(int i=0;i<4;i++) for(int j=0;j<8;j++) be[31-(8*i+j)]=(uint8_t)(x[i]>>(8*j)); BN_bin2bn(be,32,b); }

static void *lane_main(void *arg){
    t_lane = (int)(long)arg;
    BIGNUM *bP=0,*bX=0,*bI=0,*bT=0; BN_CTX *ctx=0;
    if(t_lane==0){ bP=BN_new(); bX=BN_new(); bI=BN_new(); bT=BN_new(); ctx=BN_CTX_new(); BN_hex2bn(&bP,PHEX); }
    for(long long i=0;i<N_CASES;i++){
        uint64_t x[4]; make_input(i,x);
        uint64_t Rb[5]={x[0],x[1],x[2],x[3],0};
        if(t_lane==0){ g_shfl=0; g_fneg=0; }
        warp_barrier();
        by_inverse(Rb,t_lane);
        long long by_shfl=0; uint32_t by_fneg=0;
        if(t_lane==0){ by_shfl=g_shfl; by_fneg=g_fneg; }
        uint64_t Rs[5]={x[0],x[1],x[2],x[3],0};
        st_inverse(Rs,t_lane);
        if(t_lane==0){
            memcpy(g_by,Rb,40); memcpy(g_st,Rs,40);
            g_fneg=by_fneg;
            long long nb=by_shfl/12, ns=(g_shfl-by_shfl+1)/13;
            if(by_shfl%12 || nb<1 || nb>=64){ fprintf(stderr,"FATAL: BY shuffle count %lld not 12*batches\n",by_shfl); abort(); }
            if((g_shfl-by_shfl)!=13*ns-1){ fprintf(stderr,"FATAL: Stein shuffle count %lld not 13*batches-1\n",g_shfl-by_shfl); abort(); }
            g_stein_hist[ns]++;
            g_batch_hist[nb]++; if(nb>g_maxbatch) g_maxbatch=nb;
            g_fneg_cnt += (g_fneg!=0);
            /* (a) BY == Stein, bit for bit */
            int bad=0;
            for(int j=0;j<5;j++) if(g_by[j]!=g_st[j]) bad|=1;
            /* (b) x * inv(x) == 1 (mod p), or inv(0)==0 */
            bn_from_limbs(bX,x); bn_from_limbs(bI,g_by);
            if(BN_is_zero(bX)){ if(!BN_is_zero(bI)) bad|=2; g_zero_out++; }
            else{
                BN_mod_mul(bT,bX,bI,bP,ctx);
                if(!BN_is_one(bT)) bad|=2;
                /* (c) identical to OpenSSL's own inverse */
                BIGNUM *ref=BN_mod_inverse(NULL,bX,bP,ctx);
                if(!ref || BN_cmp(ref,bI)!=0) bad|=4;
                if(ref) BN_free(ref);
                if(BN_cmp(bI,bP)>=0) bad|=8;             /* canonical range */
            }
            g_checked++;
            if(bad){ g_fail++;
                if(g_fail<6){ char *h=BN_bn2hex(bX); fprintf(stderr,"FAIL(%d) x=%s\n",bad,h); OPENSSL_free(h); } }
        }
        warp_barrier();
    }
    if(t_lane==0){ BN_free(bP);BN_free(bX);BN_free(bI);BN_free(bT);BN_CTX_free(ctx); }
    return 0;
}

static long long SEARCH=0;
static void *search_main(void *arg){
    t_lane=(int)(long)arg; uint64_t s=0x51AB1EULL;
    for(long long i=0;i<SEARCH;i++){
        uint64_t x[4]; uint64_t ss=0xBEEF0000ULL+(uint64_t)i*0x9E3779B9ULL;
        for(int j=0;j<4;j++) x[j]=splitmix(&ss);
        red_p(x); if(!(x[0]|x[1]|x[2]|x[3])) x[0]=1;
        uint64_t R[5]={x[0],x[1],x[2],x[3],0};
        if(t_lane==0) g_shfl=0;
        warp_barrier();
        by_inverse(R,t_lane);
        if(t_lane==0){ long long nb=g_shfl/12; g_batch_hist[nb]++; if(nb>g_maxbatch) g_maxbatch=nb; g_checked++;
                       if(R[0]==0&&R[1]==0&&R[2]==0&&R[3]==0) g_zero_out++; }
        warp_barrier();
    }
    (void)s; return 0;
}

int main(int argc,char**argv){
    if(argc>2 && !strcmp(argv[1],"--search")){
        SEARCH=atoll(argv[2]);
        pthread_t th[QSB_LANES];
        for(long i=1;i<QSB_LANES;i++) pthread_create(&th[i],0,search_main,(void*)i);
        search_main((void*)0);
        for(int i=1;i<QSB_LANES;i++) pthread_join(th[i],0);
        printf("batch-count search over %lld random values in [1,p):\n   histogram:",g_checked);
        for(int b=0;b<64;b++) if(g_batch_hist[b]) printf(" %d:%lld",b,g_batch_hist[b]);
        double m=0; for(int b=0;b<64;b++) m+=(double)b*g_batch_hist[b];
        printf("\n   mean %.4f  max %lld  (every case terminated; zero outputs=%lld)\n",m/(double)g_checked,g_maxbatch,g_zero_out);
        return 0;
    }
    if(argc>1) N_RAND=atoll(argv[1]);
    /* ---- unit test: shipped zi_divstep30_by vs the independent recursion ---- */
    {
        long long n=0,bad=0,clamped=0; uint64_t s=12345;
        for(int32_t d=-16;d<=16;d++) for(uint32_t fl=1;fl<16;fl+=2) for(uint32_t gl=0;gl<16;gl++)
            for(int rep=0;rep<64;rep++){
                uint32_t f=(uint32_t)(splitmix(&s)&0xFFFFFFF0u)|fl, g=(uint32_t)(splitmix(&s)&0xFFFFFFF0u)|gl;
                int32_t a,b,c,e; int32_t dd=by_divstep30(d,f,g,&a,&b,&c,&e);
                int64_t u,v,q,r; int32_t rd=ref_divstep30(d,f,g,&u,&v,&q,&r);
                n++; if(d<-4||d>4) clamped++;
                if(dd!=rd||a!=u||b!=v||c!=q||e!=r){ bad++; if(bad<4) fprintf(stderr,"DIVSTEP MISMATCH d=%d f=%08x g=%08x\n",d,f,g); }
            }
        for(long long k=0;k<3000000;k++){
            int32_t d=(int32_t)((int64_t)(splitmix(&s)%2001)-1000);
            uint32_t f=(uint32_t)splitmix(&s)|1u, g=(uint32_t)splitmix(&s);
            int32_t a,b,c,e; int32_t dd=by_divstep30(d,f,g,&a,&b,&c,&e);
            int64_t u,v,q,r; int32_t rd=ref_divstep30(d,f,g,&u,&v,&q,&r);
            n++; if(d<-4||d>4) clamped++;
            if(dd!=rd||a!=u||b!=v||c!=q||e!=r){ bad++; if(bad<4) fprintf(stderr,"DIVSTEP MISMATCH d=%d f=%08x g=%08x\n",d,f,g); }
            if(llabs((long long)a)+llabs((long long)b)>(1LL<<30)||llabs((long long)c)+llabs((long long)e)>(1LL<<30)){
                bad++; fprintf(stderr,"ROW NORM > 2^30 at d=%d f=%08x g=%08x\n",d,f,g); }
        }
        printf("zi_divstep30_by unit: %lld cases vs the direct recursion, %lld with |delta|>4 (clamped), mismatches=%lld\n",n,clamped,bad);
        if(bad) g_fail++;
    }
    build_ranges(); N_CASES=cls_hi[C_NCLS-1];
    pthread_t th[QSB_LANES];
    for(long i=1;i<QSB_LANES;i++) pthread_create(&th[i],0,lane_main,(void*)i);
    lane_main((void*)0);
    for(int i=1;i<QSB_LANES;i++) pthread_join(th[i],0);
    printf("tested=%lld  failures=%lld   (BY vs Stein, x*inv==1, vs BN_mod_inverse, canonical range)\n",g_checked,(long long)g_fail);
    for(int c=0;c<C_NCLS;c++) printf("   class %-58s n=%lld\n",cls_name[c],cls_hi[c]-cls_lo[c]);
    printf("batch histogram:"); for(int b=0;b<64;b++) if(g_batch_hist[b]) printf(" %d:%lld",b,g_batch_hist[b]); printf("\n");
    double mean=0; for(int b=0;b<64;b++) mean+=(double)b*g_batch_hist[b];
    printf("mean batches %.4f   max %lld   (Bernstein-Yang 741-divstep bound = 25 batches)\n",mean/(double)g_checked,g_maxbatch);
    printf("f ended at -1 (final conditional negate taken): %lld of %lld\n",g_fneg_cnt,g_checked);
    printf("shipped Stein path, same inputs, batch histogram:"); for(int b=0;b<64;b++) if(g_stein_hist[b]) printf(" %d:%lld",b,g_stein_hist[b]); printf("\n");
    { double m=0; for(int b=0;b<64;b++) m+=(double)b*g_stein_hist[b];
      printf("   Stein mean batches %.4f\n",m/(double)g_checked); }
    printf("RESULT: %s\n", g_fail? "FAIL":"PASS");
    return g_fail?1:0;
}
