/* h0_audit.cu — correctness oracle for QSB_GATE_H0 (word-0-only gate SHA).
 *
 * Three checks, using the REAL device macros (this file includes tree.cu, so there is no
 * transcription of the round macros and therefore no drift between audit and ranked build):
 *
 *  A. word-0 identity      qsb_sha256_init_transform_pair_w0 o[0]  ==  qsb_sha256_init_transform_pair o[0]
 *     over random compressed-pubkey gate blocks built by the real qsb_gate_block, both parities,
 *     plus constructed edge x values (0, all-ones, x with 0x00/0xff limbs).
 *  B. external oracle      the same word 0  ==  first 4 bytes of OpenSSL SHA256(33-byte encoding).
 *  C. predicate identity   qsb_gate_valid_w0(w)  ==  gpu_bench_valid_words({w,0,..})
 *     EXHAUSTIVELY over all 2^32 values of word 0. Random blocks can never exercise the
 *     >=24-leading-zero boundary (it fires once in 2^24), so this half is proved by enumeration,
 *     not by sampling.
 *
 * Build: nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_GATE_H0=1 -o h0_audit h0_audit.cu -lcrypto -lm
 */
#define main qsb_candidate_main
#include "tree.cu"
#undef main
#include <vector>
#include <openssl/sha.h>

#if !QSB_GATE_H0 || !QSB_GATE_PAIR
#error "h0_audit requires -DQSB_GATE_H0=1 and QSB_GATE_PAIR=1"
#endif

/* ---- A + B: word-0 identity, and the bytes the host will check against OpenSSL ---- */
__global__ void h0_words(const uint64_t *q, const uint32_t *par, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t q1x[4], q2x[4];
    for (int k = 0; k < 4; k++) { q1x[k] = q[i*8+k]; q2x[k] = q[i*8+4+k]; }
    uint32_t p = par[i];
    uint32_t pb0[16], pb1[16], pc0[16], pc1[16];
    qsb_gate_block(pb0, q1x, p);       qsb_gate_block(pb1, q2x, p >> 1);
    qsb_gate_block(pc0, q1x, p);       qsb_gate_block(pc1, q2x, p >> 1);
    uint32_t full0[8], full1[8], w00[1], w01[1];
    qsb_sha256_init_transform_pair(full0, pb0, full1, pb1);   /* shipped, all 8 words */
    qsb_sha256_init_transform_pair_w0(w00, pc0, w01, pc1);    /* new, word 0 only     */
    out[i*4+0] = full0[0];
    out[i*4+1] = w00[0];
    out[i*4+2] = full1[0];
    out[i*4+3] = w01[0];
}

/* ---- C: exhaustive predicate identity over all 2^32 word-0 values ---- */
__global__ void h0_pred(unsigned long long *bad, unsigned long long *hits) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i0 = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nbad = 0, nhit = 0;
    for (unsigned long long v = i0; v < (1ull << 32); v += stride) {
        uint32_t w = (uint32_t)v;
        uint32_t hs[8] = {w, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        int a = gpu_bench_valid_words(hs);
        int b = qsb_gate_valid_w0(w);
        if (a != b) nbad++;
        if (b) nhit++;
    }
    if (nbad) atomicAdd(bad, nbad);
    atomicAdd(hits, nhit);
}

#define CUDA(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));return 2;}}while(0)

/* Rebuild the 33-byte compressed encoding exactly as qsb_gate_block does, for the OpenSSL oracle. */
static void encode33(unsigned char *o, const uint64_t *qx, uint32_t parity) {
    o[0] = (unsigned char)(0x2 + (parity & 1u));
    for (int k = 0; k < 32; k++) o[1 + k] = (unsigned char)(qx[3 - (k >> 3)] >> (56 - 8 * (k & 7)));
}

int main() {
    const int N = 1 << 20;                    /* 1,048,576 cases = 2,097,152 compressions */
    std::vector<uint64_t> q(8 * (size_t)N);
    std::vector<uint32_t> par(N);
    /* xorshift64* so the fixture is reproducible without touching the harness RNG */
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; };
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 8; k++) q[(size_t)i*8+k] = rnd();
        par[i] = (uint32_t)(rnd() & 3u);
        /* constructed edges, cycled so every combination of both streams appears */
        switch (i % 11) {
            case 1: for (int k=0;k<4;k++) q[(size_t)i*8+k] = 0ull; break;
            case 2: for (int k=0;k<4;k++) q[(size_t)i*8+k] = ~0ull; break;
            case 3: for (int k=0;k<4;k++) q[(size_t)i*8+4+k] = 0ull; break;
            case 4: for (int k=0;k<4;k++) q[(size_t)i*8+4+k] = ~0ull; break;
            case 5: for (int k=0;k<8;k++) q[(size_t)i*8+k] = 0ull; break;
            case 6: for (int k=0;k<8;k++) q[(size_t)i*8+k] = ~0ull; break;
            case 7: for (int k=0;k<8;k++) q[(size_t)i*8+k] = 0x00000000FFFFFFFFull; break;
            case 8: for (int k=0;k<8;k++) q[(size_t)i*8+k] = 0xFFFFFFFF00000000ull; break;
            case 9: for (int k=0;k<8;k++) q[(size_t)i*8+k] = (k&1)?0ull:~0ull; break;
            case 10: for (int k=0;k<8;k++) q[(size_t)i*8+k] = q[(size_t)i*8+(k&~1)]; break; /* equal streams */
            default: break;
        }
    }
    uint64_t *dq; uint32_t *dp, *dout;
    CUDA(cudaMalloc(&dq, q.size()*8)); CUDA(cudaMalloc(&dp, (size_t)N*4)); CUDA(cudaMalloc(&dout, (size_t)N*16));
    CUDA(cudaMemcpy(dq, q.data(), q.size()*8, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dp, par.data(), (size_t)N*4, cudaMemcpyHostToDevice));
    h0_words<<<(N+255)/256, 256>>>(dq, dp, dout, N);
    CUDA(cudaDeviceSynchronize());
    std::vector<uint32_t> out(4*(size_t)N);
    CUDA(cudaMemcpy(out.data(), dout, (size_t)N*16, cudaMemcpyDeviceToHost));

    unsigned long long mism_w0 = 0, mism_ssl = 0;
    for (int i = 0; i < N; i++) {
        if (out[(size_t)i*4+0] != out[(size_t)i*4+1]) mism_w0++;
        if (out[(size_t)i*4+2] != out[(size_t)i*4+3]) mism_w0++;
        unsigned char m[33], h[32];
        for (int st = 0; st < 2; st++) {
            encode33(m, &q[(size_t)i*8 + 4*st], par[i] >> st);
            SHA256(m, 33, h);
            uint32_t ref = ((uint32_t)h[0]<<24)|((uint32_t)h[1]<<16)|((uint32_t)h[2]<<8)|(uint32_t)h[3];
            if (ref != out[(size_t)i*4 + 1 + 2*st]) mism_ssl++;   /* the NEW word-0 value vs OpenSSL */
        }
    }
    printf("A. word-0 identity (new vs shipped pair): %d cases x 2 streams = %d compressions, mismatches = %llu\n",
           N, 2*N, mism_w0);
    printf("B. OpenSSL oracle (new word 0 vs SHA256(33B)[0:4]): %d compressions, mismatches = %llu\n",
           2*N, mism_ssl);

    unsigned long long *dbad, *dhits, bad = 0, hits = 0;
    CUDA(cudaMalloc(&dbad, 8)); CUDA(cudaMalloc(&dhits, 8));
    CUDA(cudaMemset(dbad, 0, 8)); CUDA(cudaMemset(dhits, 0, 8));
    h0_pred<<<4096, 256>>>(dbad, dhits);
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(&hits, dhits, 8, cudaMemcpyDeviceToHost));
    printf("C. predicate identity, EXHAUSTIVE over all 2^32 word-0 values: disagreements = %llu, "
           "passing values = %llu (expected %llu for QSB_ZEROS_N=%d)\n",
           bad, hits, (unsigned long long)1 << (32 - QSB_ZEROS_N), QSB_ZEROS_N);

    int ok = (mism_w0 == 0) && (mism_ssl == 0) && (bad == 0) &&
             (hits == ((unsigned long long)1 << (32 - QSB_ZEROS_N)));
    printf("%s\n", ok ? "H0 AUDIT PASS" : "H0 AUDIT FAIL");
    return ok ? 0 : 1;
}
