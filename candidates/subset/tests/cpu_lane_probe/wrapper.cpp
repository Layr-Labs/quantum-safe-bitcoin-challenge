#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#define SIG_PUSH_SIZE 10
#define QSB_SE_PER_EPOCH 128
#define QSB_SE_TWIN 3
#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
typedef struct {
    uint32_t n, t;
    uint32_t total_preimage_len;
    uint32_t tail_section_len;
    uint32_t tx_suffix_len;
    uint32_t prefix_remainder_len;   /* NEW: bytes of fixed_prefix not in midstate */
    uint32_t midstate[8];
    uint8_t *prefix_remainder;       /* NEW: the up-to-63 bytes before dummy sigs */
    uint8_t *dummy_sigs;
    uint8_t *tail_section;
    uint8_t *tx_suffix;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} digest_params_t;

static int load_digest_params(const char *fn, digest_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(&p->n, 4, 1, f) != 1) goto err;
    if (fread(&p->t, 4, 1, f) != 1) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tail_section_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tx_suffix_len, 4, 1, f) != 1) goto err;
    if (fread(&p->prefix_remainder_len, 4, 1, f) != 1) goto err;
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i=0;i<8;i++){
        uint8_t *b=(uint8_t*)&p->midstate[i];
        p->midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (p->prefix_remainder_len > 0) {
        p->prefix_remainder = (uint8_t*)malloc(p->prefix_remainder_len);
        if (fread(p->prefix_remainder, 1, p->prefix_remainder_len, f) != p->prefix_remainder_len) goto err;
    } else {
        p->prefix_remainder = NULL;
    }
    p->dummy_sigs = (uint8_t*)malloc(p->n * SIG_PUSH_SIZE);
    if (fread(p->dummy_sigs, 1, p->n * SIG_PUSH_SIZE, f) != p->n * SIG_PUSH_SIZE) goto err;
    p->tail_section = (uint8_t*)malloc(p->tail_section_len);
    if (fread(p->tail_section, 1, p->tail_section_len, f) != p->tail_section_len) goto err;
    p->tx_suffix = (uint8_t*)malloc(p->tx_suffix_len);
    if (fread(p->tx_suffix, 1, p->tx_suffix_len, f) != p->tx_suffix_len) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    printf("  Loaded: n=%u, t=%u, preimage=%u, tail=%u, suffix=%u, prefix_rem=%u\n",
           p->n, p->t, p->total_preimage_len, p->tail_section_len, p->tx_suffix_len,
           p->prefix_remainder_len);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn); fclose(f); return -1;
}
static uint64_t binom_u64(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    __uint128_t r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
        if (r > (__uint128_t)0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFFFFFFFFFULL;
    }
    return (uint64_t)r;
}

/* Host twins of unrank_combo / qsb_rank_lex (lexicographic k-subsets of [0,n)). */
static void qsb_host_unrank(uint64_t rank, int n, int t, uint8_t *out) {
    int lo = 0;
    for (int i = 0; i < t; i++) {
        int c = lo;
        for (;;) { uint64_t cnt = binom_u64(n - c - 1, t - i - 1); if (rank < cnt) break; rank -= cnt; c++; }
        out[i] = (uint8_t)c; lo = c + 1;
    }
}

#include "../gpu_epochs/qsb_host_verify.h"
#include <chrono>
#include <sys/stat.h>
#ifndef QSB_CPU_DIAG_EPOCH
#define QSB_CPU_DIAG_EPOCH 0
#endif
#ifdef QSB_CPU_DEVBENCH
#error "This common wrapper owns timing; compile without QSB_CPU_DEVBENCH"
#endif
#ifndef CPU_HEADER
#define CPU_HEADER "../../CpuGrindSubset.h"
#endif
#include CPU_HEADER
int main(int argc,char **argv) {
 if(argc<2 || argc>3) return 2;
 const int seconds=argc==3?atoi(argv[2]):20; if(seconds<1 || seconds>120) return 2;
 mkdir("results",0755);
 digest_params_t dp={}; if(load_digest_params(argv[1],&dp)<0) return 1;
 uint8_t win3[128][3]; int cnt=0;
 // Exact promoted GPU window membership; only membership affects the CPU complement.
 for(int a=0;a<13;a++)for(int b=a+1;b<13;b++)for(int c=b+1;c<13;c++){
  if(!((a>=6)||(a<=5 && b>=7)||(a==0 && b==1 && c>=8 && c<=10)))continue;
  if(cnt>=128)return 3; win3[cnt][0]=137+a;win3[cnt][1]=137+b;win3[cnt][2]=137+c;cnt++;
 }
 if(cnt!=128)return 3;
 qcpu::start(&dp,win3,128,137,6);
 // Both lanes publish cand only after finishing a batch. No lane-specific setup
 // timestamp or changed worker instrumentation is needed for the common screen.
 using Clock=std::chrono::steady_clock;
 const auto setup_deadline=Clock::now()+std::chrono::seconds(120);
 while(!qcpu::g_ctx || qcpu::g_ctx->cand.load()==0) {
   if(Clock::now()>=setup_deadline){fprintf(stderr,"SETUP_TIMEOUT: no completed CPU batch\n");return 5;}
   usleep(10000);
 }
 sleep(1);
 const auto begin=Clock::now(); const auto before=qcpu::g_ctx->cand.load();
 sleep(seconds);
 const auto after=qcpu::g_ctx->cand.load(); const auto end=Clock::now();
 const double elapsed=std::chrono::duration<double>(end-begin).count();
 {std::lock_guard<std::mutex> lock(qcpu::g_ctx->io);
  if(qcpu::g_ctx->out)fflush(qcpu::g_ctx->out);
  printf("DEVBENCH cpu %.6f M/s (%llu candidates, %.6f s, %d workers)\n",
         (double)(after-before)/elapsed/1e6,(unsigned long long)(after-before),elapsed,qcpu::g_ctx->nthreads);
  fflush(stdout);
  // Production workers are detached; avoid global teardown racing their writes.
  _Exit(0);
 }
}
