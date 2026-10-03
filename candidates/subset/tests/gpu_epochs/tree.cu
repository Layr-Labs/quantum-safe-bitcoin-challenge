/* qsb_digest_search.cu : Multi-GPU digest round search
 *
 * Reads digest_rN.bin, enumerates C(130,9) combinations.
 * CPU generates combo batches, GPU hashes + EC recovery + 4 DER checks.
 *
 * Build:  nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm
 * Usage:  ./qsb_digest <digest_rN.bin> <gpu_index> [easy]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "launch_budget.h"
#ifndef QSB_L2_FETCH64
#define QSB_L2_FETCH64 1
#endif





#ifndef QSB_GATHER_ALL_FETCH64
#define QSB_GATHER_ALL_FETCH64 0
#endif
#if QSB_GATHER_ALL_FETCH64 < 0 || QSB_GATHER_ALL_FETCH64 > 2
#error "QSB_GATHER_ALL_FETCH64 must be 0, 1 or 2"
#endif
#if QSB_GATHER_ALL_FETCH64 == 1
#define QSB_GATHER_FETCH_NC ".nc"
#else
#define QSB_GATHER_FETCH_NC ""
#endif




#ifndef QSB_GROUP_CAP_EXACT
#define QSB_GROUP_CAP_EXACT 1
#endif




#ifndef QSB_GROUP_CAP_TIGHT
#define QSB_GROUP_CAP_TIGHT 0
#endif
static size_t qsb_group_capacity(int cut, int early, size_t epochs) {
#if QSB_GROUP_CAP_TIGHT
if(cut==137 && early==6){
if(epochs==1048576)return 181498;
if(epochs==2097152)return 362053;
if(epochs==4194304)return 594292;
}
#endif
#if QSB_GROUP_CAP_EXACT
if (cut == 137 && early == 6 && epochs == 1048576)
return 228771;
#endif
return 2 * epochs + 4;
}





#ifndef ZLAB_HITPATH
#define ZLAB_HITPATH 1
#endif




#ifndef QSB_HV_STATS
#define QSB_HV_STATS 0
#endif
#ifndef QSB_HOST_VERIFY
#define QSB_HOST_VERIFY 1
#endif
#ifndef QSB_TABLE_BASE_A
#define QSB_TABLE_BASE_A 1
#endif







#ifndef QSB_LOCAL_SM86
#define QSB_LOCAL_SM86 0



#endif
#ifndef QSB_S3
#define QSB_S3 1
#endif



#ifndef QSB_YOFF_S
#define QSB_YOFF_S 0
#endif
#if QSB_YOFF_S != 0 && QSB_YOFF_S != 1
#error "QSB_YOFF_S must be 0 or 1"
#endif
#if QSB_YOFF_S && !QSB_S3
#error "QSB_YOFF_S requires the S3 chain; replay chains use plain ordinates"
#endif
#if QSB_S3 != 0 && QSB_S3 != 1
#error "QSB_S3 must be 0 or 1"
#endif
#if QSB_S3 && !QSB_TABLE_BASE_A
#error "QSB_S3 builds its table on base A itself (z*A = r1*A + psi(r2*A)); it needs QSB_TABLE_BASE_A"
#endif
#ifndef QSB_TRIM_DIRECT_PRODUCER
#define QSB_TRIM_DIRECT_PRODUCER 1
#endif








#ifndef QSB_SLOT_PIPELINE
#define QSB_SLOT_PIPELINE 1
#endif


#ifndef QSB_FIRST_SLOT_INTERLEAVE
#define QSB_FIRST_SLOT_INTERLEAVE 0
#endif





#ifndef QSB_HOST_PRODUCERS
#define QSB_HOST_PRODUCERS 1
#endif



#ifndef QSB_HP_FIRST_ONLY_UPLOAD
#define QSB_HP_FIRST_ONLY_UPLOAD 0
#endif


#ifndef QSB_TRACE_START
#define QSB_TRACE_START 0
#endif
#if QSB_TRACE_START
#include <time.h>
static double qsb_trace_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const double qsb_trace_t_start = qsb_trace_now();
#endif





#ifndef QSB_TABLE_L2_WINDOW
#define QSB_TABLE_L2_WINDOW 1
#endif
#ifndef ZLAB_TRIM
#define ZLAB_TRIM 1
#endif


#ifndef QSB_TRIM_COMBO_ALLOC
#define QSB_TRIM_COMBO_ALLOC 0
#endif



#ifndef QSB_FIRST_PRODUCER_SCRATCH
#define QSB_FIRST_PRODUCER_SCRATCH 0
#endif




#ifndef ZLAB_PAIRSHA
#define ZLAB_PAIRSHA 0
#endif
#ifndef ZLAB_DUAL_EPOCH_SHA
#define ZLAB_DUAL_EPOCH_SHA 1
#endif
#define ZLAB_HIT_REC 16
#define ZLAB_HIT_FIRST 8
#ifndef QSB_STARTUP_TRIM
#define QSB_STARTUP_TRIM 1
#endif










#ifndef QSB_STARTUP_THREADS
#define QSB_STARTUP_THREADS 1
#endif
#ifndef QSB_S1_LADDER_SELFTEST
#define QSB_S1_LADDER_SELFTEST 0
#endif
#if QSB_S3 && !QSB_HOST_VERIFY
#error "QSB_S3 has no exact GPU chain; hits are published through the exact host gate (QSB_HOST_VERIFY=1)"
#endif
#if QSB_S3 && !QSB_STARTUP_TRIM
#error "QSB_S3: the untrimmed startup mirrors the whole (9.8 GB) table to the host for its spot check"
#endif
#include <cuda_runtime.h>

#ifndef QSB_PRODUCER_CENSUS
#define QSB_PRODUCER_CENSUS 0
#endif
#if QSB_PRODUCER_CENSUS
#include "producer_census.h"
#endif
#include <openssl/sha.h>
#include "../../QsbCarrier.h"
#include "digest_phase_audit.cuh"

#include "../../GPUMath.h"

#define MAX_LEN_WORD_PRIME 20
#define MAX_LEN_WORD_AFFIX 4
#define AFFIX_IS_SUFFIX true
#define SIZE_COMBO_MULTI 4
#define COUNT_COMBO_SYMBOLS 100
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)

__device__ __constant__ int MULTI_EIGHT[65] = { 0,
0+8,0+16,0+24,0+32,0+40,0+48,0+56,0+64,
64+8,64+16,64+24,64+32,64+40,64+48,64+56,64+64,
128+8,128+16,128+24,128+32,128+40,128+48,128+56,128+64,
192+8,192+16,192+24,192+32,192+40,192+48,192+56,192+64,
256+8,256+16,256+24,256+32,256+40,256+48,256+56,256+64,
320+8,320+16,320+24,320+32,320+40,320+48,320+56,320+64,
384+8,384+16,384+24,384+32,384+40,384+48,384+56,384+64,
448+8,448+16,448+24,448+32,448+40,448+48,448+56,448+64,
};
__device__ __constant__ uint8_t COMBO_SYMBOLS[100] = {
0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,
0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,0x40,0x5B,0x5C,0x5D,0x5E,0x5F,0x60,0x7B,0x7C,0x7D,0x7E,
0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,
0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,
0x00,0x7F,0xFF,0x09,0x0D
};

#define ASSEMBLY_SIGMA 1
#include "../../GPUHash.h"






































#ifndef QSB_SHA_SCHED_V4
#define QSB_SHA_SCHED_V4 1
#endif
#if QSB_SHA_SCHED_V4 != 0 && QSB_SHA_SCHED_V4 != 1
#error "QSB_SHA_SCHED_V4 must be 0 or 1"
#endif
#ifndef QSB_OUTER_LITK
#define QSB_OUTER_LITK 0
#endif
#if QSB_OUTER_LITK != 0 && QSB_OUTER_LITK != 1
#error "QSB_OUTER_LITK must be 0 or 1"
#endif
#ifndef QSB_S3_NM_MASK
#if QSB_LOCAL_SM86
#define QSB_S3_NM_MASK 0
#else
#define QSB_S3_NM_MASK 1
#endif
#endif


#ifndef QSB_S3_DOFF
#define QSB_S3_DOFF 1
#endif
#ifndef QSB_S3_UNIFORM_G
#define QSB_S3_UNIFORM_G 1
#endif
#if QSB_S3_UNIFORM_G != 0 && QSB_S3_UNIFORM_G != 1
#error "QSB_S3_UNIFORM_G must be 0 or 1"
#endif
#if QSB_S3_DOFF != 0 && QSB_S3_DOFF != 1
#error "QSB_S3_DOFF must be 0 or 1"
#endif
#if QSB_S3_NM_MASK != 0 && QSB_S3_NM_MASK != 1
#error "QSB_S3_NM_MASK must be 0 or 1"
#endif
#ifndef QSB_SEED_K32_SUB
#define QSB_SEED_K32_SUB 0
#endif
#if QSB_SEED_K32_SUB != 0 && QSB_SEED_K32_SUB != 1
#error "QSB_SEED_K32_SUB must be 0 or 1"
#endif
#ifndef QSB_GATHER_L1_POLICY
#define QSB_GATHER_L1_POLICY 0
#endif
#if QSB_GATHER_L1_POLICY < 0 || QSB_GATHER_L1_POLICY > 3
#error "QSB_GATHER_L1_POLICY must be 0 to 3"
#endif
#if QSB_GATHER_ALL_FETCH64 == 2 && QSB_GATHER_L1_POLICY
#error "coherent ALL_FETCH64=2 requires ordinary load policy"
#endif
#ifndef QSB_GATHER_ONE_FORM
#if QSB_LOCAL_SM86
#define QSB_GATHER_ONE_FORM 0
#else
#define QSB_GATHER_ONE_FORM 1
#endif
#endif
#if QSB_GATHER_ONE_FORM < 0 || QSB_GATHER_ONE_FORM > 2
#error "QSB_GATHER_ONE_FORM must be 0, 1 or 2"
#endif
#if (QSB_GATHER_L1_POLICY || QSB_GATHER_ONE_FORM) && !QSB_S3_NM_MASK
#error "QSB_GATHER_L1_POLICY and QSB_GATHER_ONE_FORM are written in qsb_s3_load_n (QSB_S3_NM_MASK 1)"
#endif










#ifndef QSB_SHA_CONST_IV
#define QSB_SHA_CONST_IV 1
#endif
#ifndef QSB_SHA_CONST_PEEL
#define QSB_SHA_CONST_PEEL 0
#endif
#ifndef QSB_GATE_W8_LEA
#define QSB_GATE_W8_LEA 1
#endif
#if (QSB_SHA_CONST_IV != 0 && QSB_SHA_CONST_IV != 1) || (QSB_SHA_CONST_PEEL != 0 && QSB_SHA_CONST_PEEL != 1) || \
(QSB_GATE_W8_LEA != 0 && QSB_GATE_W8_LEA != 1)
#error "QSB_SHA_CONST_IV, QSB_SHA_CONST_PEEL and QSB_GATE_W8_LEA are 0 or 1"
#endif
#if QSB_SHA_CONST_IV && !QSB_SHA_SCHED_V4
#error "QSB_SHA_CONST_IV reads the 16 B-aligned rows of QSB_SHA_SCHED_V4"
#endif
#if QSB_SHA_CONST_PEEL && !QSB_SHA_CONST_IV
#error "QSB_SHA_CONST_PEEL is written in the QSB_SHA_CONST_IV loop"
#endif

#if QSB_SHA_SCHED_V4
__device__ __constant__ __align__(16) uint32_t QSB_CONST_SCHEDULE[4][64];
#else
__device__ __constant__ uint32_t QSB_CONST_SCHEDULE[4][64];
#endif
__device__ __constant__ uint64_t QSB_U2R[8];




__device__ __constant__ uint64_t QSB_U2R_ISO[8];
__device__ __constant__ uint64_t QSB_ISO_INVU[4];
__device__ __constant__ uint32_t QSB_ISO_XNEG;





#ifndef QSB_K2S_CENTER_FUSED
#define QSB_K2S_CENTER_FUSED 0
#endif
#ifndef QSB_K2S_SUM_BASIS
#define QSB_K2S_SUM_BASIS 0
#endif
#ifndef QSB_K2S_CENTER_SQR
#define QSB_K2S_CENTER_SQR 0
#endif
#if QSB_K2S_CENTER_FUSED && QSB_K2S_CENTER_SQR != 2
#error "center fused requires qualified mode2 tail"
#endif
#if QSB_K2S_CENTER_SQR < 0 || QSB_K2S_CENTER_SQR > 2
#error "QSB_K2S_CENTER_SQR must be 0 (off), 1 (early), or 2 (after first parity)"
#endif
__device__ __constant__ uint64_t QSB_U2R_C[4 + 4*(QSB_K2S_CENTER_SQR != 0) + (QSB_K2S_CENTER_FUSED == 2)];

__device__ uint4 QSB_PUSH_WORDS[151];
static int qsb_prepare_push_words(const uint8_t *bytes,int n){
if(n<0 || n>151)return 1;
uint4 words[151];
for(int i=0;i<n;i++){
const uint8_t *r=bytes+10*i;
words[i].x=((uint32_t)r[0]<<24)|((uint32_t)r[1]<<16)|((uint32_t)r[2]<<8)|r[3];
words[i].y=((uint32_t)r[4]<<24)|((uint32_t)r[5]<<16)|((uint32_t)r[6]<<8)|r[7];
words[i].z=((uint32_t)r[8]<<8)|r[9];
words[i].w=0;
}
return QSB_TO_SYMBOL(QSB_PUSH_WORDS,words,n*sizeof(uint4))==cudaSuccess?0:1;
}
static uint32_t qsb_host_rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
static int qsb_prepare_constant_schedule(const uint32_t *words,int count){
if(count!=69)return 1;
uint32_t round_k[64],expanded[4][64];
if(QSB_FROM_SYMBOL(round_k,K,sizeof(round_k))!=cudaSuccess)return 1;
for(int block=0;block<4;block++){
uint32_t *w=expanded[block];memcpy(w,words+5+block*16,64);
for(int i=16;i<64;i++){
uint32_t x=w[i-15],y=w[i-2];
uint32_t lo=qsb_host_rotr(x,7)^qsb_host_rotr(x,18)^(x>>3);
uint32_t hi=qsb_host_rotr(y,17)^qsb_host_rotr(y,19)^(y>>10);
w[i]=w[i-16]+lo+w[i-7]+hi;
}
for(int i=0;i<64;i++)w[i]+=round_k[i];
}
return QSB_TO_SYMBOL(QSB_CONST_SCHEDULE,expanded,sizeof(expanded))==cudaSuccess?0:1;
}
template<int block> __device__ __forceinline__ void qsb_compress_constant(uint32_t *output){
uint32_t a=output[0],b=output[1],c=output[2],d=output[3],e=output[4],f=output[5],g=output[6],h=output[7],t1,t2;
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][0]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][1]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][2]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][3]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][4]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][5]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][6]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][7]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][8]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][9]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][10]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][11]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][12]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][13]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][14]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][15]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][16]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][17]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][18]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][19]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][20]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][21]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][22]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][23]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][24]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][25]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][26]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][27]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][28]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][29]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][30]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][31]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][32]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][33]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][34]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][35]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][36]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][37]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][38]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][39]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][40]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][41]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][42]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][43]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][44]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][45]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][46]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][47]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][48]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][49]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][50]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][51]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][52]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][53]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][54]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][55]);
S2Round(a, b, c, d, e, f, g, h, 0, QSB_CONST_SCHEDULE[block][56]);
S2Round(h, a, b, c, d, e, f, g, 0, QSB_CONST_SCHEDULE[block][57]);
S2Round(g, h, a, b, c, d, e, f, 0, QSB_CONST_SCHEDULE[block][58]);
S2Round(f, g, h, a, b, c, d, e, 0, QSB_CONST_SCHEDULE[block][59]);
S2Round(e, f, g, h, a, b, c, d, 0, QSB_CONST_SCHEDULE[block][60]);
S2Round(d, e, f, g, h, a, b, c, 0, QSB_CONST_SCHEDULE[block][61]);
S2Round(c, d, e, f, g, h, a, b, 0, QSB_CONST_SCHEDULE[block][62]);
S2Round(b, c, d, e, f, g, h, a, 0, QSB_CONST_SCHEDULE[block][63]);
output[0]+=a;output[1]+=b;output[2]+=c;output[3]+=d;output[4]+=e;output[5]+=f;output[6]+=g;output[7]+=h;
}


__device__ __forceinline__ void qsb_compress_constant_rolled(uint32_t *output){
#pragma unroll 1
for(int block=0;block<4;block++){
uint32_t a=output[0],b=output[1],c=output[2],d=output[3];
uint32_t e=output[4],f=output[5],g=output[6],h=output[7],t1,t2;
#pragma unroll 1
for(int r=0;r<64;r+=8){
S2Round(a,b,c,d,e,f,g,h,0,QSB_CONST_SCHEDULE[block][r]);
S2Round(h,a,b,c,d,e,f,g,0,QSB_CONST_SCHEDULE[block][r+1]);
S2Round(g,h,a,b,c,d,e,f,0,QSB_CONST_SCHEDULE[block][r+2]);
S2Round(f,g,h,a,b,c,d,e,0,QSB_CONST_SCHEDULE[block][r+3]);
S2Round(e,f,g,h,a,b,c,d,0,QSB_CONST_SCHEDULE[block][r+4]);
S2Round(d,e,f,g,h,a,b,c,0,QSB_CONST_SCHEDULE[block][r+5]);
S2Round(c,d,e,f,g,h,a,b,0,QSB_CONST_SCHEDULE[block][r+6]);
S2Round(b,c,d,e,f,g,h,a,0,QSB_CONST_SCHEDULE[block][r+7]);
}
output[0]+=a;output[1]+=b;output[2]+=c;output[3]+=d;
output[4]+=e;output[5]+=f;output[6]+=g;output[7]+=h;
}
}






__device__ uint64_t BINOM_C[151][10];


































#ifndef ZLAB_T14
#define ZLAB_T14 0
#endif
#if QSB_S3 && ZLAB_T14
#error "QSB_S3 and ZLAB_T14 are alternative table geometries"
#endif
#if QSB_S3














#ifndef QSB_GLV_FALLBACK_INLINE
#define QSB_GLV_FALLBACK_INLINE 1
#endif





#ifndef QSB_GLV11_P18
#define QSB_GLV11_P18 1
#endif
#ifndef QSB_GLV11
#if QSB_LOCAL_SM86
#define QSB_GLV11 0
#else
#define QSB_GLV11 1
#endif
#endif
#if QSB_GLV11 && !QSB_GLV11_P18
#error "this tree carries only the P18 layout of QSB_GLV11"
#endif






#ifndef QSB_Q_P18
#if QSB_LOCAL_SM86
#define QSB_Q_P18 0
#else
#define QSB_Q_P18 1
#endif
#endif
#if QSB_Q_P18 && !QSB_GLV11
#error "QSB_Q_P18 needs the GLV11 table (segments 6 and 7)"
#endif








#ifndef QSB_Q_MIX
#if QSB_LOCAL_SM86
#define QSB_Q_MIX 0
#else
#define QSB_Q_MIX 4
#endif
#endif
#if QSB_Q_MIX < 0 || (QSB_Q_MIX & (QSB_Q_MIX - 1)) != 0
#error "QSB_Q_MIX must be 0 or a power of two"
#endif
#if QSB_Q_MIX && !QSB_Q_P18
#error "QSB_Q_MIX mixes the GLV12 Q layout into the QSB_Q_P18 chain"
#endif





#ifndef QSB_Q_SPREAD
#define QSB_Q_SPREAD 0
#endif
#if QSB_Q_SPREAD != 0 && QSB_Q_SPREAD != 1
#error "QSB_Q_SPREAD must be 0 or 1"
#endif
#if QSB_Q_SPREAD && QSB_Q_MIX != 4
#error "QSB_Q_SPREAD is written for QSB_Q_MIX 4"
#endif
























#ifndef QSB_PSI_HOIST
#define QSB_PSI_HOIST 0
#endif
#if QSB_PSI_HOIST < 0 || QSB_PSI_HOIST > 11 || (QSB_PSI_HOIST & 12) == 12
#error "QSB_PSI_HOIST: bits 0, 1 and one of the form bits 2, 3"
#endif























#ifndef QSB_TREE_WAVE_TOP
#define QSB_TREE_WAVE_TOP 1
#endif
#if QSB_TREE_WAVE_TOP != 0 && QSB_TREE_WAVE_TOP != 1
#error "QSB_TREE_WAVE_TOP must be 0 or 1"
#endif

















#ifndef QSB_ROOT_LUT_SMEM
#define QSB_ROOT_LUT_SMEM 0
#endif
#if QSB_ROOT_LUT_SMEM != 0 && QSB_ROOT_LUT_SMEM != 1
#error "QSB_ROOT_LUT_SMEM must be 0 or 1"
#endif






















#ifndef QSB_PRE3_ROOT
#define QSB_PRE3_ROOT 0
#endif
#if QSB_PRE3_ROOT < 0 || QSB_PRE3_ROOT > 2
#error "QSB_PRE3_ROOT must be 0, 1 or 2"
#endif
#if QSB_PRE3_ROOT && !QSB_TREE_WAVE_TOP
#error "QSB_PRE3_ROOT hooks the wave-top branch of the tree (QSB_TREE_WAVE_TOP 1)"
#endif










#ifndef QSB_ROOT_WARP
#define QSB_ROOT_WARP 0
#endif
#if QSB_ROOT_WARP < 0 || QSB_ROOT_WARP > 7
#error "QSB_ROOT_WARP must be a warp of the 256-thread block (0..7)"
#endif
#if QSB_ROOT_WARP && !(QSB_TREE_WAVE_TOP && (QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT))
#error "QSB_ROOT_WARP needs the wave top and the templated tree (QSB_ROOT_LUT_SMEM or QSB_PRE3_ROOT)"
#endif
#if QSB_ROOT_WARP && QSB_PRE3_ROOT == 2
#error "QSB_ROOT_WARP is written for QSB_PRE3_ROOT 0 or 1"
#endif


#ifndef QSB_ROOT_PARK_B
#define QSB_ROOT_PARK_B 0
#endif
#if QSB_ROOT_PARK_B != 0 && QSB_ROOT_PARK_B != 1
#error "QSB_ROOT_PARK_B must be 0 or 1"
#endif
#if QSB_ROOT_PARK_B && (!QSB_TREE_WAVE_TOP || QSB_ROOT_WARP || QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_LUT32 || QSB_ROOT_LUT40)
#error "QSB_ROOT_PARK_B requires the original wave-top warp-0 separate inverse arena"
#endif
#ifndef QSB_DEN_CROSS_PRE
#define QSB_DEN_CROSS_PRE 0
#endif
#if QSB_DEN_CROSS_PRE < 0 || QSB_DEN_CROSS_PRE > 3
#error "QSB_DEN_CROSS_PRE must be 0 (off), 1 (both), 2 (B only), or 3 (A only)"
#endif
#if QSB_DEN_CROSS_PRE && (QSB_PRE3_ROOT || QSB_TAIL_STAGGER || QSB_TAIL_WEAVE || QSB_ROOT_PARK_B)
#error "QSB_DEN_CROSS_PRE requires the original unsplit paired tail"
#endif


#ifndef QSB_DEN_CROSS_IDLE
#define QSB_DEN_CROSS_IDLE 0
#endif
#if QSB_DEN_CROSS_IDLE && (QSB_DEN_CROSS_PRE != 1 || !QSB_TREE_WAVE_TOP || QSB_ROOT_WARP || QSB_ROOT_LUT_SMEM || QSB_ROOT_PARK_B)
#error "QSB_DEN_CROSS_IDLE requires the original wave-top tree and both relocated factors"
#endif
#if QSB_K2S_CENTER_SQR && (QSB_TAIL_STAGGER || QSB_TAIL_WEAVE)
#error "QSB_K2S_CENTER_SQR is written for the original unsplit post3 tail"
#endif



#ifndef QSB_GATE_REC_ROLL
#define QSB_GATE_REC_ROLL 0
#endif
#if QSB_GATE_REC_ROLL != 0 && QSB_GATE_REC_ROLL != 1
#error "QSB_GATE_REC_ROLL must be 0 or 1"
#endif



#ifndef QSB_GATE_REC_CALL
#define QSB_GATE_REC_CALL 0
#endif
#if QSB_GATE_REC_CALL != 0 && QSB_GATE_REC_CALL != 1
#error "QSB_GATE_REC_CALL must be 0 or 1"
#endif
#if QSB_GATE_REC_CALL && QSB_GATE_REC_ROLL
#error "Outlined and rolled gates are separate experiments"
#endif




#ifndef QSB_OUTER_FRONT_CALL
#define QSB_OUTER_FRONT_CALL 0
#endif
#if QSB_OUTER_FRONT_CALL != 0 && QSB_OUTER_FRONT_CALL != 1
#error "QSB_OUTER_FRONT_CALL must be 0 or 1"
#endif



#ifndef QSB_FRONT3_WORDS_ONLY
#define QSB_FRONT3_WORDS_ONLY 0
#endif


#ifndef QSB_FRONT3_PUBLISH_AB
#define QSB_FRONT3_PUBLISH_AB 0
#endif
#if QSB_FRONT3_PUBLISH_AB < 0 || QSB_FRONT3_PUBLISH_AB > 2
#error "QSB_FRONT3_PUBLISH_AB must be 0..2"
#endif
#ifndef QSB_FRONT3_PUBLISH_A
#define QSB_FRONT3_PUBLISH_A 0
#endif
#if QSB_FRONT3_PUBLISH_A != 0 && QSB_FRONT3_PUBLISH_A != 1
#error "QSB_FRONT3_PUBLISH_A must be 0 or 1"
#endif
#if QSB_FRONT3_WORDS_ONLY != 0 && QSB_FRONT3_WORDS_ONLY != 1
#error "QSB_FRONT3_WORDS_ONLY must be 0 or 1"
#endif







































#ifndef QSB_TAIL_STAGGER
#define QSB_TAIL_STAGGER 0
#endif
#if QSB_TAIL_STAGGER < 0 || QSB_TAIL_STAGGER > 6
#error "QSB_TAIL_STAGGER must be 0 to 6"
#endif
#ifndef QSB_TAIL_PARK
#define QSB_TAIL_PARK 4
#endif
#if QSB_TAIL_PARK < 0 || QSB_TAIL_PARK > 4
#error "QSB_TAIL_PARK must be 0 to 4"
#endif













#ifndef QSB_YNEG_FOLD
#define QSB_YNEG_FOLD 1
#endif
#if QSB_YNEG_FOLD < 0 || QSB_YNEG_FOLD > 1
#error "QSB_YNEG_FOLD must be 0 or 1"
#endif











#ifndef QSB_HIT_NO_COMBO
#define QSB_HIT_NO_COMBO 1
#endif
#if QSB_HIT_NO_COMBO < 0 || QSB_HIT_NO_COMBO > 1
#error "QSB_HIT_NO_COMBO must be 0 or 1"
#endif
#if QSB_HIT_NO_COMBO && !QSB_HOST_VERIFY
#error "QSB_HIT_NO_COMBO drops the combo bytes the GPU verify path (QSB_HOST_VERIFY=0) would copy; set it to 0 there"
#endif
















#ifndef QSB_OK_FOLD
#define QSB_OK_FOLD 2
#endif
#ifndef QSB_XNEG_BRANCH
#define QSB_XNEG_BRANCH 1
#endif
#ifndef QSB_PW_QN
#define QSB_PW_QN 0
#endif
#ifndef QSB_TID_UNSIGNED
#define QSB_TID_UNSIGNED 1
#endif










#ifndef QSB_S3_NM_SEED
#if QSB_LOCAL_SM86
#define QSB_S3_NM_SEED 0
#else
#define QSB_S3_NM_SEED 1
#endif
#endif
#ifndef QSB_TREE_UNROLL
#define QSB_TREE_UNROLL 0
#endif




#ifndef QSB_PARK128
#define QSB_PARK128 1
#endif






#ifndef QSB_R_CBANK_TAILS
#define QSB_R_CBANK_TAILS 0
#endif
#if QSB_R_CBANK_TAILS < 0 || QSB_R_CBANK_TAILS > 1
#error "QSB_R_CBANK_TAILS must be 0 or 1"
#endif






















#ifndef QSB_TAIL_WEAVE
#define QSB_TAIL_WEAVE 0
#endif
#if QSB_TAIL_WEAVE < 0 || QSB_TAIL_WEAVE > 3
#error "QSB_TAIL_WEAVE must be 0 to 3"
#endif
#if QSB_TAIL_WEAVE && !QSB_TAIL_STAGGER
#error "QSB_TAIL_WEAVE weaves the split tail calls; it needs QSB_TAIL_STAGGER"
#endif
#define QSB_WEAVE_PARK_STRIDE 2048








#ifndef QSB_DIVSTEP_LOOKAHEAD
#define QSB_DIVSTEP_LOOKAHEAD 0
#endif
#if QSB_DIVSTEP_LOOKAHEAD < 0 || QSB_DIVSTEP_LOOKAHEAD > 1
#error "QSB_DIVSTEP_LOOKAHEAD must be 0 or 1"
#endif
#include "../../GLVScalar.cuh"
#if QSB_GLV11
#define GT_CHUNKS 8
#define GT_TOTAL_ENTRIES 354501773u
#define GT_GLV_TERMS (QSB_Q_P18 ? 10 : 11)
#else
#define GT_CHUNKS 6
#define GT_GLV_TERMS 12
#define GT_TOTAL_ENTRIES 153175181u
#endif

#define GT_LO 4096
#define GT_HI 4096
#define GT_H2 16
#define GT_DENSE_ENTRIES 786432u


#ifndef QSB_GT_HEAL
#define QSB_GT_HEAL 1
#endif
#if QSB_GT_HEAL != 0 && QSB_GT_HEAL != 1
#error "QSB_GT_HEAL must be 0 or 1"
#endif
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
#if QSB_GLV11
if(c>=6) return c==6 ? 67108864u : 134217728u;
#endif
return q9_bigtbl_entries(c);
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
#if QSB_GLV11
if(c>=6) return c==6 ? 153175181u : 220284045u;
#endif
return q9_bigtbl_offset(c);
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
#if QSB_GLV11
if(c>=6) return c==6 ? 18 : 45;
#endif
return (int)q9_bigtbl_shift(c);
}
static_assert(GT_TOTAL_ENTRIES*64ULL == (QSB_GLV11 ? 22688113472ULL : 9803211584ULL),
"GLV12 table must contain exactly 9,803,211,584 bytes");
static_assert(GT_TOTAL_ENTRIES < 0x80000000u, "record index must not use the sign bit");
static_assert(262144u+262144u+131072u+131072u == GT_DENSE_ENTRIES &&
GT_DENSE_ENTRIES*64ULL == (48ULL<<20),
"segments 0-3 must be the 48 MiB pinned prefix");
static_assert(GT_DENSE_ENTRIES+67108864u+85279885u+(QSB_GLV11 ? 201326592u : 0u) == GT_TOTAL_ENTRIES,
"segments 4 and 5 must end the table");
static_assert(((2u*67108864u-1u)>>24) < GT_H2 && ((2u*85279885u-1u)>>24) < GT_H2,
"H2 ladder must cover the largest odd multiplier");
#if QSB_GLV11
static_assert(((2u*134217728u-1u)>>24) < GT_H2, "P18 high ladder must cover m=2^28-1");
#endif
#elif ZLAB_T14
#define GT_CHUNKS 14
#define GT_BIG 4
#define GT_TOTAL_ENTRIES (GT_BIG * (1u << 18) + (GT_CHUNKS - GT_BIG) * (1u << 17))
#define GT_LO 256
#define GT_HI 2048
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
return c < GT_BIG ? (1u << 18) : (1u << 17);
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
return c <= GT_BIG ? (unsigned)c << 18 : ((unsigned)GT_BIG << 18) + ((unsigned)(c - GT_BIG) << 17);
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
return c <= GT_BIG ? 19*c : 19*GT_BIG + 18*(c - GT_BIG);
}
static_assert(GT_TOTAL_ENTRIES*64ULL == 144ULL*1024*1024,
"14-term table must contain exactly 144 MiB");
#else
#define GT_CHUNKS 15
#define GT_TOTAL_ENTRIES (1u << 20)
#define GT_LO 256
#define GT_HI 1024
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
return c == 0 ? (1u << 17) : (1u << 16);
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
return c == 0 ? 0u : (unsigned)(c+1) << 16;
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
return c == 0 ? 0 : 17*c+1;
}
static_assert(GT_TOTAL_ENTRIES*64ULL == 64ULL*1024*1024,
"mixed table must contain exactly 64 MiB");
#endif


__device__ __constant__ uint64_t GT_ORDER_N[4] = {
0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};









__device__ __forceinline__ void gt_recode_setup(const uint64_t k[4], uint64_t M[4], int *sign) {
const uint64_t n0=GT_ORDER_N[0], n1=GT_ORDER_N[1], n2=GT_ORDER_N[2], n3=GT_ORDER_N[3];
__uint128_t s;



uint64_t k0=k[0], k1=k[1], k2=k[2], k3=k[3];
if (k3 == n3 &&
(k2 > n2 ||
(k2 == n2 && (k1 > n1 || (k1 == n1 && k0 >= n0))))) {
s=(__uint128_t)k0-n0; k0=(uint64_t)s; uint64_t kb=(uint64_t)(s>>64)&1;
s=(__uint128_t)k1-n1-kb; k1=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
s=(__uint128_t)k2-n2-kb; k2=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
s=(__uint128_t)k3-n3-kb; k3=(uint64_t)s;
}
#if QSB_TABLE_BASE_A
uint64_t m0=k0, m1=k1, m2=k2, m3=k3;
uint64_t br;
#else
uint64_t t0=k0<<1;
uint64_t t1=(k1<<1)|(k0>>63);
uint64_t t2=(k2<<1)|(k1>>63);
uint64_t t3=(k3<<1)|(k2>>63);
uint64_t tc=(k3>>63);
s=(__uint128_t)t0-n0; uint64_t d0=(uint64_t)s; uint64_t br=(s>>64)&1;
s=(__uint128_t)t1-n1-br; uint64_t d1=(uint64_t)s; br=(s>>64)&1;
s=(__uint128_t)t2-n2-br; uint64_t d2=(uint64_t)s; br=(s>>64)&1;
s=(__uint128_t)t3-n3-br; uint64_t d3=(uint64_t)s; br=(s>>64)&1;
uint64_t ge = tc | (1u - (uint64_t)br);
uint64_t gm = 0 - ge;
uint64_t m0=(t0&~gm)|(d0&gm), m1=(t1&~gm)|(d1&gm), m2=(t2&~gm)|(d2&gm), m3=(t3&~gm)|(d3&gm);
#endif
uint64_t odd = m0 & 1ULL;
s=(__uint128_t)n0-m0; uint64_t p0=(uint64_t)s; br=(s>>64)&1;
s=(__uint128_t)n1-m1-br; uint64_t p1=(uint64_t)s; br=(s>>64)&1;
s=(__uint128_t)n2-m2-br; uint64_t p2=(uint64_t)s; br=(s>>64)&1;
s=(__uint128_t)n3-m3-br; uint64_t p3=(uint64_t)s;
uint64_t om = 0 - odd;
M[0]=(m0&om)|(p0&~om); M[1]=(m1&om)|(p1&~om); M[2]=(m2&om)|(p2&~om); M[3]=(m3&om)|(p3&~om);
*sign = (int)odd*2 - 1;
}

template<int BITS>
__device__ __forceinline__ int32_t gt_mixed_step(uint64_t M[4], int sign) {
int32_t digit=(int32_t)(M[0]&((1u<<(BITS+1))-1))-(1<<BITS);
uint64_t r0=(M[0]>>(BITS+1))|(M[1]<<(63-BITS));
uint64_t r1=(M[1]>>(BITS+1))|(M[2]<<(63-BITS));
uint64_t r2=(M[2]>>(BITS+1))|(M[3]<<(63-BITS));
uint64_t r3=M[3]>>(BITS+1);
M[0]=(r0<<1)|1ULL; M[1]=(r1<<1)|(r0>>63);
M[2]=(r2<<1)|(r1>>63); M[3]=(r3<<1)|(r2>>63);
return sign*digit;
}
#if !QSB_S3
__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4], int32_t e[GT_CHUNKS]) {
uint64_t M[4]; int sign; gt_recode_setup(k,M,&sign);
#if ZLAB_T14
#pragma unroll
for(int c=0;c<GT_BIG;c++)e[c]=gt_mixed_step<19>(M,sign);
#pragma unroll
for(int c=GT_BIG;c<GT_CHUNKS-1;c++)e[c]=gt_mixed_step<18>(M,sign);
#else
e[0]=gt_mixed_step<18>(M,sign);
#pragma unroll
for(int c=1;c<GT_CHUNKS-1;c++)e[c]=gt_mixed_step<17>(M,sign);
#endif
e[GT_CHUNKS-1]=sign*(int32_t)M[0];
}
#endif



__device__ __forceinline__ void gt_load_signed_flat(const uint8_t *__restrict__ gTable,
uint32_t base, uint32_t idx,
uint64_t neg,
uint64_t *__restrict__ gx,
uint64_t *__restrict__ gy) {
size_t off = ((size_t)base + idx) * 64;
const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
ulonglong2 x0=__ldg(tx),x1=__ldg(tx+1),y0=__ldg(ty),y1=__ldg(ty+1);
gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
uint64_t m=0ULL-neg;
uint64_t r0=y0.x^m, r1=y0.y^m, r2=y1.x^m, r3=y1.y^m;
uint64_t c0=0xFFFFFFFEFFFFFC30ULL&m;
UADDO1(r0,c0); UADDC1(r1,m); UADDC1(r2,m); UADD1(r3,m);
gy[0]=r0; gy[1]=r1; gy[2]=r2; gy[3]=r3;
}

__device__ __forceinline__ void gt_load_signed(const uint8_t *gTable,
int c, uint32_t idx, uint64_t neg,
uint64_t gx[4], uint64_t gy[4]) {
gt_load_signed_flat(gTable, gt_offset(c), idx, neg, gx, gy);
}
#ifndef QSB_NEG_SHORT
#define QSB_NEG_SHORT 1
#endif





__device__ __forceinline__ void gt_load_signed_flat_f(const uint8_t *__restrict__ gTable,
uint32_t base, uint32_t idx, uint64_t neg,
uint64_t *__restrict__ gx,
uint64_t *__restrict__ gy) {
#if QSB_NEG_SHORT
size_t off = ((size_t)base + idx) * 64;
const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
ulonglong2 x0=__ldg(tx),x1=__ldg(tx+1),y0=__ldg(ty),y1=__ldg(ty+1);
gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
uint64_t m=0ULL-neg;
gy[0]=(y0.x^m)+(0xFFFFFFFEFFFFFC30ULL&m); gy[1]=y0.y^m; gy[2]=y1.x^m; gy[3]=y1.y^m;
#else
gt_load_signed_flat(gTable, base, idx, neg, gx, gy);
#endif
}





__device__ __forceinline__ void gt_digit_idx(int32_t ec, uint32_t *idx, uint64_t *neg) {
int32_t mask = ec >> 31;
uint32_t ae = ((uint32_t)ec ^ (uint32_t)mask) - (uint32_t)mask;
*idx = (ae - 1) >> 1;
*neg = (uint64_t)(mask & 1);
}
































#ifndef ZLAB_DIRDIG
#define ZLAB_DIRDIG 1
#endif
#if ZLAB_DIRDIG
__device__ __forceinline__ uint32_t gt_field_bits_v(const uint64_t m[4], unsigned pos) {
unsigned li = pos >> 6, sh = pos & 63u;
uint64_t lo = li == 0 ? m[0] : li == 1 ? m[1] : li == 2 ? m[2] : m[3];
uint64_t hi = li == 0 ? m[1] : li == 1 ? m[2] : li == 2 ? m[3] : 0ULL;
return (uint32_t)((lo >> sh) | ((hi << 1) << (63u - sh)));
}

__host__ __device__ __forceinline__ unsigned gt_width(int c) {
return c == GT_CHUNKS-1 ? (unsigned)(gt_shift(c) - gt_shift(c-1))
: (unsigned)(gt_shift(c+1) - gt_shift(c));
}
__device__ __forceinline__ void gt_direct_digit(const uint64_t M[4], uint64_t sflag,
unsigned pos, unsigned w, bool last,
uint32_t *idx, uint64_t *neg) {
uint32_t f = gt_field_bits_v(M, pos) & ((1u << w) - 1u);
uint32_t t = f >> (w - 1);
*idx = last ? (f & ((1u << (w - 1)) - 1u)) : ((f ^ (t - 1u)) & ((1u << (w - 1)) - 1u));
*neg = (last ? 0ULL : (uint64_t)(t ^ 1u)) ^ sflag;
}
#endif




__device__ __forceinline__ void qsb_double_affine(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
const uint64_t *x,const uint64_t *y){
uint64_t yy[4],yyyy[4],ss[4],mm[4],tt[4],uu[4];
_ModSqr(yy,(uint64_t*)y);_ModSqr(yyyy,yy);
_ModMult(ss,(uint64_t*)x,yy);_ModAdd256(ss,ss,ss);_ModAdd256(ss,ss,ss);
_ModSqr(mm,(uint64_t*)x);_ModAdd256(tt,mm,mm);_ModAdd256(mm,mm,tt);
_ModSqr(tt,mm);_ModSub256(tt,tt,ss);_ModSub256(X,tt,ss);
_ModSub256(uu,ss,X);_ModMult(uu,mm);
_ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);
_ModSub256(Y,uu,yyyy);
_ModAdd256(ZZ,yy,yy);_ModAdd256(ZZ,ZZ,ZZ);
_ModMult(ZZZ,(uint64_t*)y,ZZ);_ModAdd256(ZZZ,ZZZ,ZZZ);
}
__device__ __forceinline__ void qsb_complete_last_add(
uint64_t *X1,uint64_t *Y1,uint64_t *ZZ1,uint64_t *ZZZ1,
const uint64_t *X2,const uint64_t *Y2,const uint64_t *Yoff){
uint64_t U2[4],S2[4],P[4],R[4],PP[4],PPP[4],Q[4],T[4];
_ModMult(U2,(uint64_t*)X2,ZZ1);
_ModAdd256(S2,(uint64_t*)Y2,(uint64_t*)Yoff);_ModMult(S2,ZZZ1);
_ModSub256(P,U2,X1);_ModSub256(R,S2,Y1);
if(!(P[0]|P[1]|P[2]|P[3])){
if(!(R[0]|R[1]|R[2]|R[3])) qsb_double_affine(X1,Y1,ZZ1,ZZZ1,X2,Y2);
else {
#pragma unroll
for(int i=0;i<4;i++){X1[i]=0;Y1[i]=(i==0);ZZ1[i]=ZZZ1[i]=0;}
}
return;
}
_ModSqr(PP,P);_ModMult(PPP,PP,P);_ModMult(Q,U2,PP);_ModMult(ZZ1,PP);
_ModSqr(T,R);_ModAdd256(T,T,PPP);_ModSub256(T,T,Q);_ModSub256(T,T,Q);
_ModMult(ZZZ1,PPP);_ModSub256(Q,Q,T);_ModMult(Q,R);
_ModMult(S2,(uint64_t*)Y2,ZZZ1);_ModSub256(Y1,Q,S2);Load256(X1,T);
}

#include "../../chain_replay_field.cuh"
#include "../../hit_filter_field.cuh"
#if QSB_YOFF_S && (QSB_SC_ALUZ || !QSB_SHORT_CARRY || !QSB_Y_PAIR || !QSB_NEG_SHORT || QSB_SC_PP)
#error "QSB_YOFF_S requires rolled SC Y-pair and signed short-load path"
#endif
#include "filter_tail_sc.cuh"



#ifndef QSB_SPEC_LAST_RESOLVE
#define QSB_SPEC_LAST_RESOLVE 1
#endif
__device__ __forceinline__ void qsb_filter_last_add(
uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
const uint64_t *x,const uint64_t *y,uint64_t *yoff,uint32_t &bad, uint64_t *RP = nullptr) {
#if QSB_Y_PAIR


qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x,y,yoff,bad,RP);
uint64_t scaled_y[4];
#if QSB_YOFF_S




const uint64_t yy[4] = {y[0] - 0x800001E8ULL, y[1], y[2], y[3]};
qsb_filter_fused16(scaled_y,yy,ZZZ,Y,RP);
#else
qsb_filter_fused16(scaled_y,y,ZZZ,Y,RP);
#endif
#if QSB_YNEG_FOLD

Load256(Y, scaled_y);
return;
#endif
{ const uint64_t zero[4] = {0ULL, 0ULL, 0ULL, 0ULL}; Load256(Y, zero); }
#else
#if QSB_YNEG_FOLD
#error "QSB_YNEG_FOLD folds the QSB_Y_PAIR resolve (0 - s); set it to 0 without QSB_Y_PAIR"
#endif
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x,y,yoff,bad);
uint64_t scaled_y[4];
qsb_filter_mul(scaled_y,y,ZZZ,bad);
#endif
#if QSB_SPEC_LAST_RESOLVE
QSB_FSUB(Y,Y,scaled_y);
#else
_ModSub256(Y,Y,scaled_y);
#endif
}
#if !QSB_S3



__device__ void qsb_replay_chain_exact(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
const uint64_t k[4], const uint8_t *gTable) {
uint64_t M[4]; int sign;
gt_recode_setup(k, M, &sign);
uint32_t idx; uint64_t neg;
uint64_t x0[4],y0[4],x1[4],y1[4];
#if ZLAB_T14
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed(gTable,0,idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed(gTable,1,idx,neg,x1,y1);
#else
int32_t ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
#endif
_PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#if ZLAB_DIRDIG
unsigned pos=(unsigned)gt_shift(2)+1u;
#endif
#pragma unroll 1
for (int c=2;c<GT_BIG;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg); pos+=gt_width(2);
#else
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
_PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
Load256(y0, cy);
table_base += 1u << 18;
}
#pragma unroll 1
for (int c=GT_BIG;c<GT_CHUNKS-1;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),false,&idx,&neg); pos+=gt_width(GT_BIG);
#else
ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
_PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
Load256(y0, cy);
table_base += 1u << 17;
}
{
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),true,&idx,&neg);
#else
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#else
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed(gTable,0,idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed(gTable,1,idx,neg,x1,y1);
_PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
unsigned pos=(unsigned)gt_shift(2)+1u;
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg);
pos+=gt_width(2);
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
_PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
Load256(y0, cy);
table_base += 1u << 16;
}
{
gt_direct_digit(M,sflag,pos,gt_width(2),true,&idx,&neg);
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#else
int32_t ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
_PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
_PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
Load256(y0, cy);
table_base += 1u << 16;
}
{
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#endif
#endif
}
__device__ void qsb_replay_chain_trial(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
const uint64_t k[4], const uint8_t *gTable, uint32_t &bad) {
uint64_t M[4]; int sign;
gt_recode_setup(k, M, &sign);
uint32_t idx; uint64_t neg;
uint64_t x0[4],y0[4],x1[4],y1[4];
#if ZLAB_T14
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed(gTable,0,idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed(gTable,1,idx,neg,x1,y1);
#else
int32_t ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
#endif
qsb_replay_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#if ZLAB_DIRDIG
unsigned pos=(unsigned)gt_shift(2)+1u;
#endif
#pragma unroll 1
for (int c=2;c<GT_BIG;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg); pos+=gt_width(2);
#else
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_replay_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 18;
}
#pragma unroll 1
for (int c=GT_BIG;c<GT_CHUNKS-1;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),false,&idx,&neg); pos+=gt_width(GT_BIG);
#else
ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_replay_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 17;
}
{
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),true,&idx,&neg);
#else
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#else
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed(gTable,0,idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed(gTable,1,idx,neg,x1,y1);
qsb_replay_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
unsigned pos=(unsigned)gt_shift(2)+1u;
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg);
pos+=gt_width(2);
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_replay_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 16;
}
{
gt_direct_digit(M,sflag,pos,gt_width(2),true,&idx,&neg);
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#else
int32_t ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
qsb_replay_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_replay_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 16;
}
{
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_complete_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
}
#endif
#endif
}
#endif
#ifndef QSB_DIGIT_SHIFT
#define QSB_DIGIT_SHIFT 1
#endif
#ifndef QSB_CHAIN_UNROLL
#define QSB_CHAIN_UNROLL 1
#endif
#if QSB_S3














typedef struct { uint32_t mask, centre, off, width; } qsb_s3_desc_t;
#if QSB_Q_P18
#define QSB_S3_PSI_TERM 5
#else
#define QSB_S3_PSI_TERM 6
#endif
#if QSB_GLV11
#define QSB_S3_DESC_G11 { \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFu,1u<<19,262144u,19u}, \
{0x3FFFFu,1u<<18,524288u,18u}, \
{0x3FFFFu,1u<<18,655360u,18u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}, \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFFFu,1u<<27,153175181u,27u}, \
{0xFFFFFFFu,1u<<28,220284045u,28u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}}
#endif
#if QSB_GLV11 && QSB_Q_P18
#define QSB_S3_DESC_INIT { \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFFFu,1u<<27,153175181u,27u}, \
{0xFFFFFFFu,1u<<28,220284045u,28u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}, \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFFFu,1u<<27,153175181u,27u}, \
{0xFFFFFFFu,1u<<28,220284045u,28u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}}
#elif QSB_GLV11
#define QSB_S3_DESC_INIT QSB_S3_DESC_G11
#else
#define QSB_S3_DESC_INIT { \
{0x3FFFFu, 0u, 0u, 18u}, \
{0x7FFFFu, 1u << 19, 262144u, 19u}, \
{0x3FFFFu, 1u << 18, 524288u, 18u}, \
{0x3FFFFu, 1u << 18, 655360u, 18u}, \
{0x7FFFFFFu, 1u << 27, 786432u, 27u}, \
{0xFFFFFFFu, 170559770u, 67895296u, 28u}, \
{0x3FFFFu, 0u, 0u, 18u}, \
{0x7FFFFu, 1u << 19, 262144u, 19u}, \
{0x3FFFFu, 1u << 18, 524288u, 18u}, \
{0x3FFFFu, 1u << 18, 655360u, 18u}, \
{0x7FFFFFFu, 1u << 27, 786432u, 27u}, \
{0xFFFFFFFu, 170559770u, 67895296u, 28u}}
#endif
__device__ __constant__ qsb_s3_desc_t QSB_S3_DESC[GT_GLV_TERMS] = QSB_S3_DESC_INIT;
#if QSB_Q_MIX



#define QSB_S3_MXF_PTAIL 11u
#define QSB_S3_MXF_LAST 15u
#define QSB_S3_DESC_MXF_INIT { \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFFFu,1u<<27,153175181u,27u}, \
{0xFFFFFFFu,1u<<28,220284045u,28u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}, \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFu,1u<<19,262144u,19u}, \
{0x3FFFFu,1u<<18,524288u,18u}, \
{0x3FFFFu,1u<<18,655360u,18u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}, \
{0x3FFFFu,0u,0u,18u}, \
{0x7FFFFFFu,1u<<27,153175181u,27u}, \
{0xFFFFFFFu,1u<<28,220284045u,28u}, \
{0x7FFFFFFu,1u<<27,786432u,27u}, \
{0xFFFFFFFu,170559770u,67895296u,28u}}
__device__ __constant__ qsb_s3_desc_t QSB_S3_DESC_MXF[QSB_S3_MXF_LAST + 1u] = QSB_S3_DESC_MXF_INIT;
#endif



#ifndef QSB_S3_HALF_DIGIT
#define QSB_S3_HALF_DIGIT 0
#endif
#if QSB_S3_HALF_DIGIT != 0 && QSB_S3_HALF_DIGIT != 1
#error "QSB_S3_HALF_DIGIT must be 0 or 1"
#endif



#ifndef QSB_S3_BORROW_DIGIT
#define QSB_S3_BORROW_DIGIT 0
#endif
#if QSB_S3_BORROW_DIGIT != 0 && QSB_S3_BORROW_DIGIT != 1
#error "QSB_S3_BORROW_DIGIT must be 0 or 1"
#endif



#ifndef QSB_S3_SELECT_DIGIT
#define QSB_S3_SELECT_DIGIT 0
#endif
#if QSB_S3_SELECT_DIGIT != 0 && QSB_S3_SELECT_DIGIT != 1
#error "QSB_S3_SELECT_DIGIT must be 0 or 1"
#endif



#ifndef QSB_S3_ABSDIFF_DIGIT
#define QSB_S3_ABSDIFF_DIGIT 0
#endif
#if QSB_S3_ABSDIFF_DIGIT < 0 || QSB_S3_ABSDIFF_DIGIT > 3
#error "QSB_S3_ABSDIFF_DIGIT must be 0 (off), 1 (unsigned video), 2 (signed video), or 3 (signed scalar)"
#endif
#if QSB_S3_BORROW_DIGIT + QSB_S3_HALF_DIGIT + QSB_S3_SELECT_DIGIT + (QSB_S3_ABSDIFF_DIGIT != 0) > 1
#error "borrow, half-digit, predicate-select and absdiff representations are exclusive"
#endif
#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
__host__ __device__ __forceinline__ uint32_t qsb_s3_borrow_half(uint32_t f, uint32_t threshold, uint32_t &nm) {
#ifdef __CUDA_ARCH__
uint32_t h;
#if QSB_S3_SELECT_DIGIT

asm("{.reg .pred p; .reg .u32 hh;\n\t"
"sub.u32 hh, %2, %3;\n\tsetp.lt.u32 p, %2, %3;\n\t"
"selp.u32 %1, 0xffffffff, 0, p;\n\tmov.u32 %0, hh;}"
: "=r"(h), "=r"(nm) : "r"(f), "r"(threshold));
#else
asm("sub.cc.u32 %0, %2, %3;\n\tsubc.u32 %1, 0, 0;"
: "=r"(h), "=r"(nm) : "r"(f), "r"(threshold));
#endif
return h;
#else
nm = 0u - (uint32_t)(f < threshold);
return f - threshold;
#endif
}
#endif
#if QSB_S3_ABSDIFF_DIGIT
__host__ __device__ __forceinline__ uint32_t qsb_s3_absdiff_index(uint32_t f, uint32_t threshold,
uint32_t off, uint32_t &nm) {
#ifdef __CUDA_ARCH__
uint32_t idx;
asm("{.reg .pred p; .reg .u32 sum, mask;\n\t"
"setp.lt.u32 p, %2, %3;\n\t"
"selp.u32 mask, 0xffffffff, 0, p;\n\t"
#if QSB_S3_ABSDIFF_DIGIT == 3
"sad.s32 sum, %2, %3, %4;\n\t"
#elif QSB_S3_ABSDIFF_DIGIT == 2


"vabsdiff.s32.s32.s32.add sum, %2, %3, %4;\n\t"
#else
"vabsdiff.u32.u32.u32.add sum, %2, %3, %4;\n\t"
#endif
"add.u32 %0, sum, mask;\n\tmov.u32 %1, mask;}"
: "=r"(idx), "=r"(nm) : "r"(f), "r"(threshold), "r"(off));
return idx;
#else
nm = 0u - (uint32_t)(f < threshold);
return (f < threshold ? threshold - f - 1u : f - threshold) + off;
#endif
}
#endif
#if QSB_GLV_ZDEC
#if !QSB_Q_MIX || !QSB_GLV11 || !QSB_Q_P18
#error "QSB_GLV_ZDEC is written for the QSB_Q_MIX walk (GLV11, Q_P18)"
#endif



#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT || QSB_S3_ABSDIFF_DIGIT
#define QSB_S3_ZD(mask, centre, off, width) {mask, (centre) >> 1, off, width}
#define QSB_S3_Z0(m) ((m) & (1u << 18))
#elif QSB_S3_HALF_DIGIT
#define QSB_S3_ZD(mask, centre, off, width) {mask, 0u - ((centre) >> 1), off, width}
#define QSB_S3_Z0(m) ((m) << 18)
#else
#define QSB_S3_ZD(mask, centre, off, width) {mask, 1u - (centre), off, width}
#define QSB_S3_Z0(m) (((m) << 19) | 1u)
#endif
#define QSB_S3_ZDESC_MXF_INIT { \
QSB_S3_ZD(0x3FFFFu,0u,0u,18u), \
QSB_S3_ZD(0x7FFFFFFu,1u<<27,153175181u,27u), \
QSB_S3_ZD(0xFFFFFFFu,1u<<28,220284045u,28u), \
QSB_S3_ZD(0x7FFFFFFu,1u<<27,786432u,27u), \
QSB_S3_ZD(0xFFFFFFFu,170559770u,67895296u,28u), \
QSB_S3_ZD(0x3FFFFu,0u,0u,18u), \
QSB_S3_ZD(0x7FFFFu,1u<<19,262144u,19u), \
QSB_S3_ZD(0x3FFFFu,1u<<18,524288u,18u), \
QSB_S3_ZD(0x3FFFFu,1u<<18,655360u,18u), \
QSB_S3_ZD(0x7FFFFFFu,1u<<27,786432u,27u), \
QSB_S3_ZD(0xFFFFFFFu,170559770u,67895296u,28u), \
QSB_S3_ZD(0x3FFFFu,0u,0u,18u), \
QSB_S3_ZD(0x7FFFFFFu,1u<<27,153175181u,27u), \
QSB_S3_ZD(0xFFFFFFFu,1u<<28,220284045u,28u), \
QSB_S3_ZD(0x7FFFFFFu,1u<<27,786432u,27u), \
QSB_S3_ZD(0xFFFFFFFu,170559770u,67895296u,28u)}
__device__ __constant__ qsb_s3_desc_t QSB_S3_ZDESC_MXF[QSB_S3_MXF_LAST + 1u] = QSB_S3_ZDESC_MXF_INIT;
#endif
typedef struct { uint32_t w[8], signs; } qsb_s3_walker;
__host__ __device__ __forceinline__ uint32_t qsb_s3_shr(uint32_t lo, uint32_t hi, uint32_t s) {
#ifdef __CUDA_ARCH__
return __funnelshift_r(lo, hi, s);
#else
return (uint32_t)(((((uint64_t)hi) << 32) | lo) >> (s & 31u));
#endif
}


__host__ __device__ __forceinline__ void qsb_s3_begin(qsb_s3_walker &w,
const uint64_t magP[2], unsigned sP, const uint64_t magQ[2], unsigned sQ) {
w.w[0] = (uint32_t)magQ[0]; w.w[1] = (uint32_t)(magQ[0] >> 32);
w.w[2] = (uint32_t)magQ[1]; w.w[3] = (uint32_t)(magQ[1] >> 32);
w.w[4] = (uint32_t)magP[0]; w.w[5] = (uint32_t)(magP[0] >> 32);
w.w[6] = (uint32_t)magP[1]; w.w[7] = (uint32_t)(magP[1] >> 32);
w.signs = sQ | (sP << 1);
}
__host__ __device__ __forceinline__ uint32_t qsb_s3_code(qsb_s3_walker &w, int t, const qsb_s3_desc_t d) {
const uint32_t f = w.w[0] & d.mask;
const uint32_t dig = 2u * f + 1u - d.centre;
const uint32_t nm = (uint32_t)((int32_t)dig >> 31);
const uint32_t idx = ((dig ^ nm) - nm - 1u) >> 1;
const uint32_t sign = (w.signs >> (t >= QSB_S3_PSI_TERM ? 1 : 0)) & 1u;
#pragma unroll
for (int i = 0; i < 7; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[7] >>= d.width;
return (d.off + idx) | (((nm & 1u) ^ sign) << 31);
}







#ifndef QSB_S3_HALF_WALK
#define QSB_S3_HALF_WALK 1
#endif






#ifndef QSB_S3_SIGN_SHIFT
#define QSB_S3_SIGN_SHIFT 1
#endif







#ifndef QSB_S3_ODD_FOLD
#define QSB_S3_ODD_FOLD 1
#endif
#if QSB_S3_ODD_FOLD != 0 && QSB_S3_ODD_FOLD != 1
#error "QSB_S3_ODD_FOLD must be 0 or 1"
#endif
__host__ __device__ __forceinline__ uint32_t qsb_s3_code_half(qsb_s3_walker &w, int t, const qsb_s3_desc_t d) {
#if QSB_S3_ABSDIFF_DIGIT
const uint32_t f = w.w[0] & d.mask;
uint32_t nm;
const uint32_t idx = qsb_s3_absdiff_index(f, d.centre >> 1, d.off, nm);
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
const uint32_t sign = (w.signs >> (QSB_S3_SIGN_SHIFT ? 0 : (t >= QSB_S3_PSI_TERM ? 1 : 0))) & 1u;
return idx | (((nm & 1u) ^ sign) << 31);
#else
const uint32_t f = w.w[0] & d.mask;
#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
uint32_t nm;
const uint32_t dig = qsb_s3_borrow_half(f, d.centre >> 1, nm);
#elif QSB_S3_HALF_DIGIT


const uint32_t dig = f - (d.centre >> 1);
#else
const uint32_t dig = 2u * f + 1u - d.centre;
#endif
#if !QSB_S3_BORROW_DIGIT && !QSB_S3_SELECT_DIGIT
const uint32_t nm = (uint32_t)((int32_t)dig >> 31);
#endif
#if QSB_S3_HALF_DIGIT || QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
static_assert(QSB_S3_ODD_FOLD, "half-digit needs odd-fold decoder");
const uint32_t idx = dig ^ nm;
#elif QSB_S3_ODD_FOLD
const uint32_t idx = (dig ^ nm) >> 1;
#else
const uint32_t idx = ((dig ^ nm) - nm - 1u) >> 1;
#endif
#if QSB_S3_SIGN_SHIFT
(void)t;
const uint32_t sign = w.signs & 1u;
#else
const uint32_t sign = (w.signs >> (t >= QSB_S3_PSI_TERM ? 1 : 0)) & 1u;
#endif
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
return (d.off + idx) | (((nm & 1u) ^ sign) << 31);
#endif
}
__host__ __device__ __forceinline__ void qsb_s3_psi_swap(qsb_s3_walker &w) {
w.w[0] = w.w[4]; w.w[1] = w.w[5]; w.w[2] = w.w[6]; w.w[3] = w.w[7];
#if QSB_S3_SIGN_SHIFT
w.signs >>= 1;
#endif
}
#if QSB_GLV_ZDEC







__host__ __device__ __forceinline__ void qsb_s3_begin_z(qsb_s3_walker &w,
const uint64_t vP[2], uint32_t mP, const uint64_t vQ[2]) {
w.w[0] = (uint32_t)vQ[0]; w.w[1] = (uint32_t)(vQ[0] >> 32);
w.w[2] = (uint32_t)vQ[1]; w.w[3] = (uint32_t)(vQ[1] >> 32);
w.w[4] = (uint32_t)vP[0]; w.w[5] = (uint32_t)(vP[0] >> 32);
w.w[6] = (uint32_t)vP[1]; w.w[7] = (uint32_t)(vP[1] >> 32);
w.signs = QSB_S3_Z0(mP);
}
__host__ __device__ __forceinline__ uint32_t qsb_s3_code_z(qsb_s3_walker &w, const qsb_s3_desc_t d) {
#if QSB_S3_ABSDIFF_DIGIT
const uint32_t f = w.w[0] & d.mask;
uint32_t nm;
const uint32_t idx = qsb_s3_absdiff_index(f, d.centre, d.off, nm);
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
return idx | (nm << 31);
#else
const uint32_t f = w.w[0] & d.mask;
#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
uint32_t nm;
const uint32_t dig = qsb_s3_borrow_half(f, d.centre, nm);
#elif QSB_S3_HALF_DIGIT
const uint32_t dig = f + d.centre;
#else
const uint32_t dig = 2u * f + d.centre;
#endif
#if !QSB_S3_BORROW_DIGIT && !QSB_S3_SELECT_DIGIT
const uint32_t nm = (uint32_t)((int32_t)dig >> 31);
#endif
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
#if QSB_S3_HALF_DIGIT || QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
return ((dig ^ nm) | (nm << 31)) + d.off;
#else
return qsb_s3_shr(dig ^ nm, nm, 1u) + d.off;
#endif
#endif
}




__host__ __device__ __forceinline__ uint32_t qsb_s3_code_zn(qsb_s3_walker &w, const qsb_s3_desc_t d,
uint32_t &nm) {
#if QSB_S3_ABSDIFF_DIGIT
const uint32_t f = w.w[0] & d.mask;
const uint32_t idx = qsb_s3_absdiff_index(f, d.centre, d.off, nm);
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
return idx;
#else
const uint32_t f = w.w[0] & d.mask;
#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
const uint32_t dig = qsb_s3_borrow_half(f, d.centre, nm);
#elif QSB_S3_HALF_DIGIT
const uint32_t dig = f + d.centre;
#else
const uint32_t dig = 2u * f + d.centre;
#endif
#if !QSB_S3_BORROW_DIGIT && !QSB_S3_SELECT_DIGIT
nm = (uint32_t)((int32_t)dig >> 31);
#endif
#pragma unroll
for (int i = 0; i < 3; i++) w.w[i] = qsb_s3_shr(w.w[i], w.w[i + 1], d.width);
w.w[3] >>= d.width;
#if QSB_S3_HALF_DIGIT || QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT
return (dig ^ nm) + d.off;
#else
return ((dig ^ nm) >> 1) + d.off;
#endif
#endif
}
__host__ __device__ __forceinline__ void qsb_s3_psi_swap_z(qsb_s3_walker &w) {
w.w[0] = w.w[4]; w.w[1] = w.w[5]; w.w[2] = w.w[6]; w.w[3] = w.w[7];
}
#if !QSB_S3_HALF_WALK || !QSB_S3_ODD_FOLD || !QSB_S3_SIGN_SHIFT || QSB_SC_PP
#error "QSB_GLV_ZDEC is written for the rolled half-walk chain (S3_HALF_WALK, S3_ODD_FOLD, S3_SIGN_SHIFT, SC_PP 0)"
#endif
#endif
#if QSB_S3_HALF_WALK
#define QSB_S3_CODE qsb_s3_code_half
#else
#define QSB_S3_CODE qsb_s3_code
#endif


#ifndef QSB_S3_FINAL_PREFETCH
#define QSB_S3_FINAL_PREFETCH 0
#endif
#if QSB_S3_FINAL_PREFETCH != 0 && QSB_S3_FINAL_PREFETCH != 1
#error "QSB_S3_FINAL_PREFETCH must be 0 or 1"
#endif
#if QSB_S3_FINAL_PREFETCH
#if QSB_SC_PP || (QSB_Q_MIX && (!QSB_S3_NM_MASK || !QSB_S3_DOFF))
#error "final prefetch requires rolled chain and mixed NM/byte-offset decoding"
#endif
__device__ __forceinline__ void qsb_s3_prefetch_idx(const uint8_t *gTable, uint32_t idx) {
asm volatile("{ .reg .u64 a,g; cvt.u64.u32 a,%0; shl.b64 a,a,6; "
"add.u64 a,a,%1; cvta.to.global.u64 g,a; "
"prefetch.global.L2 [g]; add.u64 g,g,32; prefetch.global.L2 [g]; }"
:: "r"(idx), "l"(gTable) : "memory");
}
#endif









#ifndef QSB_GATHER_LEA2
#define QSB_GATHER_LEA2 1
#endif
#if QSB_GATHER_LEA2 != 0 && QSB_GATHER_LEA2 != 1
#error "QSB_GATHER_LEA2 must be 0 or 1"
#endif
__device__ __forceinline__ void qsb_s3_load(const uint8_t *__restrict__ gTable, uint32_t code, bool ef,
uint64_t *__restrict__ gx, uint64_t *__restrict__ gy) {
#if QSB_GATHER_LEA2
uint64_t a;
asm("{\n\t.reg .u32 r;\n\t.reg .u64 w;\n\t"
"and.b32 r,%1,0x7fffffff;\n\t"
"cvt.u64.u32 w,r;\n\t"
"shl.b64 w,w,6;\n\t"
"add.u64 %0,w,%2;\n\t}"
: "=l"(a) : "r"(code), "l"((uint64_t)gTable));
const ulonglong2 *tx = (const ulonglong2 *)a;
#else
const ulonglong2 *tx = (const ulonglong2 *)(gTable + (size_t)(code & 0x7fffffffu) * 64);
#endif
const ulonglong2 *ty = tx + 2;
ulonglong2 x0, x1, y0, y1;
#if defined(QSB_CARRIER_BUILD) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750



if (ef) {
asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs.nc.L2::64B.v2.u64 {%0,%1}, [g]; }"
: "=l"(x0.x), "=l"(x0.y) : "l"(tx));
#if QSB_GATHER_ALL_FETCH64
#define QSB_S3_COLD64(v, p) asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs" QSB_GATHER_FETCH_NC ".L2::64B.v2.u64 {%0,%1}, [g]; }" \
: "=l"((v).x), "=l"((v).y) : "l"(p))
QSB_S3_COLD64(x1, tx + 1); QSB_S3_COLD64(y0, ty); QSB_S3_COLD64(y1, ty + 1);
#undef QSB_S3_COLD64
#else
x1 = __ldcs(tx + 1); y0 = __ldcs(ty); y1 = __ldcs(ty + 1);
#endif
}
#else
if (ef) { x0 = __ldcs(tx); x1 = __ldcs(tx + 1); y0 = __ldcs(ty); y1 = __ldcs(ty + 1); }
#endif
else { x0 = __ldg(tx); x1 = __ldg(tx + 1); y0 = __ldg(ty); y1 = __ldg(ty + 1); }
gx[0] = x0.x; gx[1] = x0.y; gx[2] = x1.x; gx[3] = x1.y;
const uint64_t m = 0ULL - (uint64_t)(code >> 31);
#if QSB_NEG_SHORT
#if QSB_YOFF_S
gy[0] = y0.x ^ m; gy[1] = y0.y ^ m; gy[2] = y1.x ^ m; gy[3] = y1.y ^ m;
#else
gy[0] = (y0.x ^ m) + (0xFFFFFFFEFFFFFC30ULL & m); gy[1] = y0.y ^ m; gy[2] = y1.x ^ m; gy[3] = y1.y ^ m;
#endif
#else
uint64_t r0 = y0.x ^ m, r1 = y0.y ^ m, r2 = y1.x ^ m, r3 = y1.y ^ m;
uint64_t c0 = 0xFFFFFFFEFFFFFC30ULL & m;
UADDO1(r0, c0); UADDC1(r1, m); UADDC1(r2, m); UADD1(r3, m);
gy[0] = r0; gy[1] = r1; gy[2] = r2; gy[3] = r3;
#endif
}
#if QSB_S3_NM_MASK || QSB_S3_NM_SEED
#if !QSB_GLV_ZDEC || !QSB_GATHER_LEA2 || !QSB_NEG_SHORT
#error "QSB_S3_NM_MASK is written for the ZDEC walker with QSB_GATHER_LEA2 and QSB_NEG_SHORT"
#endif




__device__ __forceinline__ void qsb_s3_load_n(const uint8_t *__restrict__ gTable, uint32_t idx, uint32_t nm,
bool ef, uint64_t *__restrict__ gx, uint64_t *__restrict__ gy) {
uint64_t a;
asm("{\n\t.reg .u64 w;\n\t"
"cvt.u64.u32 w,%1;\n\t"
"shl.b64 w,w,6;\n\t"
"add.u64 %0,w,%2;\n\t}"
: "=l"(a) : "r"(idx), "l"((uint64_t)gTable));
const ulonglong2 *tx = (const ulonglong2 *)a;
const ulonglong2 *ty = tx + 2;
ulonglong2 x0, x1, y0, y1;
#if QSB_GATHER_ONE_FORM == 1
const bool cold = true; (void)ef;
#elif QSB_GATHER_ONE_FORM == 2
const bool cold = false; (void)ef;
#else
const bool cold = ef;
#endif
#if defined(QSB_CARRIER_BUILD) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#if QSB_GATHER_L1_POLICY && __CUDA_ARCH__ >= 800




#if QSB_GATHER_L1_POLICY == 2
#define QSB_S3_L1Q ".L1::evict_first"
#else
#define QSB_S3_L1Q ".L1::no_allocate"
#endif
#define QSB_S3_LD2(v, p, q) asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.nc" QSB_S3_L1Q q \
".v2.u64 {%0,%1}, [g]; }" : "=l"((v).x), "=l"((v).y) : "l"(p))
#if QSB_GATHER_ALL_FETCH64
if (cold) { QSB_S3_LD2(x0, tx, ".L2::64B"); QSB_S3_LD2(x1, tx + 1, ".L2::64B"); QSB_S3_LD2(y0, ty, ".L2::64B"); QSB_S3_LD2(y1, ty + 1, ".L2::64B"); }
#else
if (cold) { QSB_S3_LD2(x0, tx, ".L2::64B"); QSB_S3_LD2(x1, tx + 1, ""); QSB_S3_LD2(y0, ty, ""); QSB_S3_LD2(y1, ty + 1, ""); }
#endif
#if QSB_GATHER_L1_POLICY == 3
else { QSB_S3_LD2(x0, tx, ""); QSB_S3_LD2(x1, tx + 1, ""); QSB_S3_LD2(y0, ty, ""); QSB_S3_LD2(y1, ty + 1, ""); }
#else
else { x0 = __ldg(tx); x1 = __ldg(tx + 1); y0 = __ldg(ty); y1 = __ldg(ty + 1); }
#endif
#undef QSB_S3_LD2
#undef QSB_S3_L1Q
#else
if (cold) {
asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs.nc.L2::64B.v2.u64 {%0,%1}, [g]; }"
: "=l"(x0.x), "=l"(x0.y) : "l"(tx));
#if QSB_GATHER_ALL_FETCH64
#define QSB_S3_COLD64(v, p) asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs" QSB_GATHER_FETCH_NC ".L2::64B.v2.u64 {%0,%1}, [g]; }" \
: "=l"((v).x), "=l"((v).y) : "l"(p))
QSB_S3_COLD64(x1, tx + 1); QSB_S3_COLD64(y0, ty); QSB_S3_COLD64(y1, ty + 1);
#undef QSB_S3_COLD64
#else
x1 = __ldcs(tx + 1); y0 = __ldcs(ty); y1 = __ldcs(ty + 1);
#endif
}
else { x0 = __ldg(tx); x1 = __ldg(tx + 1); y0 = __ldg(ty); y1 = __ldg(ty + 1); }
#endif
#else
if (cold) { x0 = __ldcs(tx); x1 = __ldcs(tx + 1); y0 = __ldcs(ty); y1 = __ldcs(ty + 1); }
else { x0 = __ldg(tx); x1 = __ldg(tx + 1); y0 = __ldg(ty); y1 = __ldg(ty + 1); }
#endif
gx[0] = x0.x; gx[1] = x0.y; gx[2] = x1.x; gx[3] = x1.y;
uint64_t m;
asm("mov.b64 %0, {%1,%1};" : "=l"(m) : "r"(nm));
#if QSB_YOFF_S
gy[0] = y0.x ^ m; gy[1] = y0.y ^ m; gy[2] = y1.x ^ m; gy[3] = y1.y ^ m;
#else
gy[0] = (y0.x ^ m) + (0xFFFFFFFEFFFFFC30ULL & m); gy[1] = y0.y ^ m; gy[2] = y1.x ^ m; gy[3] = y1.y ^ m;
#endif
}
#endif










#ifndef QSB_EC_PSI_ZZ
#define QSB_EC_PSI_ZZ 0
#endif
#if QSB_EC_PSI_ZZ < 0 || QSB_EC_PSI_ZZ > 2
#error "QSB_EC_PSI_ZZ must be 0, 1 or 2"
#endif
#if QSB_EC_PSI_ZZ && QSB_SC_PP
#error "QSB_EC_PSI_ZZ requires the rolled chain (SC_PP=0)"
#endif
#if QSB_EC_PSI_ZZ
__device__ __forceinline__ void qsb_filter_psi_zz(uint64_t *ZZ, uint32_t &bad) {
const uint64_t beta2[4] = {0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL,
0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL};
#if QSB_EC_PSI_ZZ == 2
qsb_filter_mul(ZZ, beta2, ZZ, bad);
#else
qsb_filter_mul(ZZ, ZZ, beta2, bad);
#endif
}
#endif
__device__ void qsb_filter_chain_trial(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
const uint64_t k[4], const uint8_t *gTable, uint32_t &bad) {
#if QSB_GLV_ZDEC
uint64_t vz[2][2]; uint32_t mz[2];
q9_glv_split_z(k, vz[0], vz[1], &mz[0], &mz[1]);
qsb_s3_walker w;
qsb_s3_begin_z(w, vz[0], mz[0], vz[1]);
#else
uint64_t mag[2][2]; unsigned sgn[2];
q9_glv_split(k, mag[0], mag[1], &sgn[0], &sgn[1]);
qsb_s3_walker w;
qsb_s3_begin(w, mag[0], sgn[0], mag[1], sgn[1]);
#endif
uint64_t x0[4], y0[4], x1[4], y1[4];
#if QSB_Q_MIX
#if !QSB_S3_HALF_WALK || !QSB_S3_SIGN_SHIFT
#error "QSB_Q_MIX needs the half walker and the sign shift (the code form must not read t)"
#endif




#if QSB_Q_SPREAD
const unsigned wib = (threadIdx.x >> 5) & 7u;
const unsigned g = (wib == 0u) | (wib == 7u);
#else
const unsigned g = (((blockIdx.x * blockDim.x + threadIdx.x) >> 5) & (QSB_Q_MIX - 1u)) == 0u;
#endif
#if QSB_S3_UNIFORM_G




const unsigned gu = __all_sync(0xffffffffu, g);
#define g gu
#endif
#if QSB_GLV_ZDEC
const qsb_s3_desc_t d0 = {0x3FFFFu, QSB_S3_Z0(mz[1]), 0u, 18u};
#if QSB_S3_NM_MASK || QSB_S3_NM_SEED
uint32_t nm;
uint32_t code = qsb_s3_code_zn(w, d0, nm);
qsb_s3_load_n(gTable, code, nm, false, x0, y0);
const qsb_s3_desc_t d1 = QSB_S3_ZDESC_MXF[5u * g + 1u];
code = qsb_s3_code_zn(w, d1, nm);
qsb_s3_load_n(gTable, code, nm, d1.off >= GT_DENSE_ENTRIES, x1, y1);
#else
uint32_t code = qsb_s3_code_z(w, d0);
qsb_s3_load(gTable, code, false, x0, y0);
const qsb_s3_desc_t d1 = QSB_S3_ZDESC_MXF[5u * g + 1u];
code = qsb_s3_code_z(w, d1);
qsb_s3_load(gTable, code, d1.off >= GT_DENSE_ENTRIES, x1, y1);
#endif
#else
uint32_t code = QSB_S3_CODE(w, 0, QSB_S3_DESC[0]);
qsb_s3_load(gTable, code, false, x0, y0);
const qsb_s3_desc_t d1 = QSB_S3_DESC_MXF[5u * g + 1u];
code = QSB_S3_CODE(w, 1, d1);
qsb_s3_load(gTable, code, d1.off >= GT_DENSE_ENTRIES, x1, y1);
#endif
uint64_t RP[4];
qsb_filter_point_seed(X, Y, ZZ, ZZZ, x0, y0, x1, y1, bad, RP);
uint64_t cx[4], cy[4];
#if QSB_SC_PP






uint64_t Xb[4], Yb[4], ZZb[4], ZZZb[4], yb[4];
uint64_t RPb[4];
unsigned i = 5u * g + 2u;
#if QSB_SC_PP == 2


#pragma unroll 1
for (;;) {
const qsb_s3_desc_t d = QSB_S3_DESC_MXF[i];
if (d.off == 0u) {
i = QSB_S3_MXF_PTAIL;
qsb_s3_psi_swap(w);
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
}
code = qsb_s3_code_half(w, 0, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, Xb, Yb, ZZb, ZZZb, yb, bad, RP, RPb);
i++;
if (i >= QSB_S3_MXF_LAST) {
Load256(X, Xb); Load256(Y, Yb); Load256(ZZ, ZZb); Load256(ZZZ, ZZZb); Load256(y0, yb);
#if QSB_Y_PAIR
Load256(RP, RPb);
#endif
break;
}
const qsb_s3_desc_t e = QSB_S3_DESC_MXF[i];
if (e.off == 0u) {
i = QSB_S3_MXF_PTAIL;
qsb_s3_psi_swap(w);
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(Xb, Xb, beta, bad);
}
code = qsb_s3_code_half(w, 0, e);
qsb_s3_load(gTable, code, e.off >= GT_DENSE_ENTRIES, cx, cy);
qsb_filter_point_add<true>(Xb, Yb, ZZb, ZZZb, cx, cy, yb, X, Y, ZZ, ZZZ, y0, bad, RPb, RP);
i++;
if (i >= QSB_S3_MXF_LAST) break;
}
#else
if (g == 0u) {
const qsb_s3_desc_t d = QSB_S3_DESC_MXF[i];
code = qsb_s3_code_half(w, 0, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, X, Y, ZZ, ZZZ, y0, bad, RP, RP);
i++;
}
#pragma unroll 1
for (;;) {
const qsb_s3_desc_t d = QSB_S3_DESC_MXF[i];
if (d.off == 0u) {
i = QSB_S3_MXF_PTAIL;
qsb_s3_psi_swap(w);
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
}
code = qsb_s3_code_half(w, 0, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, Xb, Yb, ZZb, ZZZb, yb, bad, RP, RPb);
i++;





const qsb_s3_desc_t e = QSB_S3_DESC_MXF[i];
#if QSB_SC_PP != 3
if (e.off == 0u) {
i = QSB_S3_MXF_PTAIL;
qsb_s3_psi_swap(w);
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(Xb, Xb, beta, bad);
}
#endif
code = qsb_s3_code_half(w, 0, e);
qsb_s3_load(gTable, code, e.off >= GT_DENSE_ENTRIES, cx, cy);
qsb_filter_point_add<true>(Xb, Yb, ZZb, ZZZb, cx, cy, yb, X, Y, ZZ, ZZZ, y0, bad, RPb, RP);
i++;
if (i >= QSB_S3_MXF_LAST) break;
}
#endif
#else
#pragma unroll 1
#if QSB_S3_DOFF


for (unsigned ib = (5u * g + 2u) * (unsigned)sizeof(qsb_s3_desc_t);
ib < QSB_S3_MXF_LAST * (unsigned)sizeof(qsb_s3_desc_t); ib += (unsigned)sizeof(qsb_s3_desc_t)) {
#else
for (unsigned i = 5u * g + 2u; i < QSB_S3_MXF_LAST; i++) {
#endif
#if QSB_GLV_ZDEC
#if QSB_S3_DOFF
qsb_s3_desc_t d = *(const qsb_s3_desc_t *)((const char *)QSB_S3_ZDESC_MXF + ib);
if (d.off == 0u) {
ib = QSB_S3_MXF_PTAIL * (unsigned)sizeof(qsb_s3_desc_t);
#else
qsb_s3_desc_t d = QSB_S3_ZDESC_MXF[i];
if (d.off == 0u) {
i = QSB_S3_MXF_PTAIL;
#endif
qsb_s3_psi_swap_z(w);
d.centre = w.signs;
#if QSB_EC_PSI_ZZ
qsb_filter_psi_zz(ZZ, bad);
#else
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
#endif
}
#if QSB_S3_NM_MASK
code = qsb_s3_code_zn(w, d, nm);
qsb_s3_load_n(gTable, code, nm, d.off >= GT_DENSE_ENTRIES, cx, cy);
#else
code = qsb_s3_code_z(w, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
#endif
#else
const qsb_s3_desc_t d = QSB_S3_DESC_MXF[i];
if (d.off == 0u) {
i = QSB_S3_MXF_PTAIL;
qsb_s3_psi_swap(w);
#if QSB_EC_PSI_ZZ
qsb_filter_psi_zz(ZZ, bad);
#else
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
#endif
}
code = qsb_s3_code_half(w, 0, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
#endif
#if QSB_S3_FINAL_PREFETCH
if (ib == (QSB_S3_MXF_LAST - 1u) * (unsigned)sizeof(qsb_s3_desc_t)) {

const uint32_t dig = 2u * (w.w[0] & 0xFFFFFFFu) + 170559770u;
const uint32_t mask = (uint32_t)((int32_t)dig >> 31);
qsb_s3_prefetch_idx(gTable, ((dig ^ mask) >> 1) + 67895296u);
}
#endif
qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, bad, RP);
#if !QSB_CHAIN_ANCHOR_UPDATE
Load256(y0, cy);
#endif
}
#endif
#if QSB_GLV_ZDEC
{
const qsb_s3_desc_t dl = QSB_S3_ZD(0xFFFFFFFu, 170559770u, 67895296u, 28u);
#if QSB_S3_NM_MASK || QSB_S3_NM_SEED
code = qsb_s3_code_zn(w, dl, nm);
qsb_s3_load_n(gTable, code, nm, true, cx, cy);
#else
code = qsb_s3_code_z(w, dl);
qsb_s3_load(gTable, code, true, cx, cy);
#endif
}
#else
code = QSB_S3_CODE(w, GT_GLV_TERMS - 1, QSB_S3_DESC[GT_GLV_TERMS - 1]);
qsb_s3_load(gTable, code, true, cx, cy);
#endif
qsb_filter_last_add(X, Y, ZZ, ZZZ, cx, cy, y0, bad, RP);
#if QSB_S3_UNIFORM_G
#undef g
#endif
#else
uint32_t code = QSB_S3_CODE(w, 0, QSB_S3_DESC[0]);
qsb_s3_load(gTable, code, false, x0, y0);
code = QSB_S3_CODE(w, 1, QSB_S3_DESC[1]);
qsb_s3_load(gTable, code, QSB_Q_P18 != 0, x1, y1);
uint64_t RP[4];
qsb_filter_point_seed(X, Y, ZZ, ZZZ, x0, y0, x1, y1, bad, RP);
uint64_t cx[4], cy[4];
#pragma unroll 1
for (int t = 2; t < GT_GLV_TERMS - 1; t++) {
const qsb_s3_desc_t d = QSB_S3_DESC[t];
#if QSB_S3_HALF_WALK


if (t == QSB_S3_PSI_TERM) {
qsb_s3_psi_swap(w);
#if QSB_EC_PSI_ZZ
qsb_filter_psi_zz(ZZ, bad);
#else
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
#endif
}
code = qsb_s3_code_half(w, t, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
#else
code = qsb_s3_code(w, t, d);
qsb_s3_load(gTable, code, d.off >= GT_DENSE_ENTRIES, cx, cy);
if (t == QSB_S3_PSI_TERM) {


#if QSB_EC_PSI_ZZ
qsb_filter_psi_zz(ZZ, bad);
#else
const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
qsb_filter_mul(X, X, beta, bad);
#endif
}
#endif
#if QSB_S3_FINAL_PREFETCH
if (t == GT_GLV_TERMS - 2) {
qsb_s3_walker peek = w;
const uint32_t next = QSB_S3_CODE(peek, GT_GLV_TERMS - 1, QSB_S3_DESC[GT_GLV_TERMS - 1]);
qsb_s3_prefetch_idx(gTable, next & 0x7fffffffu);
}
#endif
qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, bad, RP);
#if !QSB_CHAIN_ANCHOR_UPDATE
Load256(y0, cy);
#endif
}
code = QSB_S3_CODE(w, GT_GLV_TERMS - 1, QSB_S3_DESC[GT_GLV_TERMS - 1]);
qsb_s3_load(gTable, code, true, cx, cy);
qsb_filter_last_add(X, Y, ZZ, ZZZ, cx, cy, y0, bad, RP);
#endif
}
#else
#if QSB_Y_PAIR
#error "QSB_Y_PAIR is wired for the QSB_S3 chains only"
#endif
__device__ void qsb_filter_chain_trial(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
const uint64_t k[4], const uint8_t *gTable, uint32_t &bad) {
uint64_t M[4]; int sign;
gt_recode_setup(k, M, &sign);
uint32_t idx; uint64_t neg;
uint64_t x0[4],y0[4],x1[4],y1[4];
#if ZLAB_T14
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed(gTable,0,idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed(gTable,1,idx,neg,x1,y1);
#else
int32_t ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
#endif
qsb_filter_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#if ZLAB_DIRDIG
unsigned pos=(unsigned)gt_shift(2)+1u;
#endif
#pragma unroll 1
for (int c=2;c<GT_BIG;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg); pos+=gt_width(2);
#else
ec=gt_mixed_step<19>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 18;
}
#pragma unroll 1
for (int c=GT_BIG;c<GT_CHUNKS-1;c++){
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),false,&idx,&neg); pos+=gt_width(GT_BIG);
#else
ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 17;
}
{
#if ZLAB_DIRDIG
gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),true,&idx,&neg);
#else
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg);
#endif
gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_filter_last_add(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
}
#else
#if ZLAB_DIRDIG
uint64_t sflag=(uint64_t)(sign<0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
gt_load_signed_flat_f(gTable,gt_offset(0),idx,neg,x0,y0);
gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
gt_load_signed_flat_f(gTable,gt_offset(1),idx,neg,x1,y1);
qsb_filter_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#if QSB_DIGIT_SHIFT




constexpr unsigned P0=36u, W2=17u;

uint32_t w0,w1,w2,w3,w4,w5,w6;
{
const uint64_t S0=(M[0]>>P0)|(M[1]<<(64-P0)), S1=(M[1]>>P0)|(M[2]<<(64-P0));
const uint64_t S2=(M[2]>>P0)|(M[3]<<(64-P0)), S3=M[3]>>P0;
w0=(uint32_t)S0; w1=(uint32_t)(S0>>32); w2=(uint32_t)S1; w3=(uint32_t)(S1>>32);
w4=(uint32_t)S2; w5=(uint32_t)(S2>>32); w6=(uint32_t)S3;
}
constexpr int kChainUnroll=QSB_CHAIN_UNROLL;
#pragma unroll (kChainUnroll)
for (int c=2;c<GT_CHUNKS-1;c++){
{
const uint32_t f=w0&((1u<<W2)-1u), t=f>>(W2-1u);
idx=(f^(t-1u))&((1u<<(W2-1u))-1u);
neg=(uint64_t)(t^1u)^sflag;
}
w0=__funnelshift_r(w0,w1,W2); w1=__funnelshift_r(w1,w2,W2); w2=__funnelshift_r(w2,w3,W2);
w3=__funnelshift_r(w3,w4,W2); w4=__funnelshift_r(w4,w5,W2); w5=__funnelshift_r(w5,w6,W2); w6>>=W2;
gt_load_signed_flat_f(gTable,table_base,idx,neg,cx,cy);
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
#if !QSB_CHAIN_ANCHOR_UPDATE
Load256(y0, cy);
#endif
table_base += 1u << 16;
}
{
const uint32_t f=w0&((1u<<W2)-1u);
idx=f&((1u<<(W2-1u))-1u); neg=sflag;
gt_load_signed_flat_f(gTable,table_base,idx,neg,cx,cy);
qsb_filter_last_add(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
}
#else
unsigned pos=(unsigned)gt_shift(2)+1u;
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg);
pos+=gt_width(2);
gt_load_signed_flat_f(gTable,table_base,idx,neg,cx,cy);
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 16;
}
{
gt_direct_digit(M,sflag,pos,gt_width(2),true,&idx,&neg);
gt_load_signed_flat_f(gTable,table_base,idx,neg,cx,cy);
qsb_filter_last_add(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
}
#endif
#else
int32_t ec=gt_mixed_step<18>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
qsb_filter_point_seed(X,Y,ZZ,ZZZ, x0,y0, x1,y1,bad);
uint64_t cx[4],cy[4];
uint32_t table_base=gt_offset(2);
#pragma unroll 1
for (int c=2;c<GT_CHUNKS-1;c++){
ec=gt_mixed_step<17>(M,sign);
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_filter_point_add<true>(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
Load256(y0, cy);
table_base += 1u << 16;
}
{
ec=sign*(int32_t)M[0];
gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
qsb_filter_last_add(X,Y,ZZ,ZZZ, cx,cy, y0,bad);
}
#endif
#endif
}
#endif
#if !QSB_S3
__device__ void _FixedBaseSignedXYZZStream(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
const uint64_t k[4], const uint8_t *gTable) {

uint64_t saved_k[4];Load256(saved_k,k);
uint32_t bad=0;
qsb_replay_chain_trial(X,Y,ZZ,ZZZ,saved_k,gTable,bad);
if(bad)qsb_replay_chain_exact(X,Y,ZZ,ZZZ,saved_k,gTable);
}
#endif









__device__ int gpu_is_valid_der(const uint8_t *d, int l) {
if(l<9||d[0]!=0x30) return 0;
int tl=d[1]; if(tl+3!=l) return 0;
int idx=2;
for(int p=0;p<2;p++){
if(idx>=l-1||d[idx]!=0x02) return 0; idx++;
int il=d[idx]; idx++;
if(il==0||idx+il>l-1) return 0;
if(il>1&&d[idx]==0&&!(d[idx+1]&0x80)) return 0;
if(d[idx]&0x80) return 0; idx+=il;}
return idx==l-1;
}
__device__ int gpu_is_der_easy(const uint8_t *d, int l) { return l>=9&&(d[0]>>4)==3; }






__device__ int gpu_is_der_relaxed(const uint8_t *d, int l) {
if (l < 9) return 0;
int tl = d[1]; if (tl + 3 != l) return 0;
int idx = 2;
for (int p = 0; p < 2; p++) {
if (idx >= l - 1 || d[idx] != 0x02) return 0; idx++;
int il = d[idx]; idx++;
if (il == 0 || idx + il > l - 1) return 0;
if (il > 1 && d[idx] == 0 && !(d[idx+1] & 0x80)) return 0;
if (d[idx] & 0x80) return 0; idx += il;
}
return idx == l - 1;
}


















#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
__device__ int gpu_leading_zero_bits(const uint8_t *h) {
int z = 0;
for (int i = 0; i < 32; i++) {
if (h[i] == 0) { z += 8; continue; }
unsigned v = h[i]; int c = 0;
while ((v & 0x80u) == 0) { c++; v <<= 1; }
return z + c;
}
return z;
}



__device__ int gpu_bench_valid(const uint8_t *h) {
return gpu_leading_zero_bits(h) >= QSB_ZEROS_N;
}




#if ZLAB_PAIRSHA

#define ZP_RND2(k) { \
for (int zr = 0; zr < 16; zr++) { \
S2RoundZ(a0,b0,c0,d0,e0,f0,g0,h0,x0,K[k+zr],w0[zr]); \
S2RoundZ(a1,b1,c1,d1,e1,f1,g1,h1,x1,K[k+zr],w1[zr]); \
} }
#define S2RoundZ(a,b,c,d,e,f,g,h,x,k,w) { \
uint32_t zt1 = h + S1(e) + Ch(e,f,g) + (k) + (w); \
uint32_t zt2 = S0(a) + Maj(a,b,c); \
d += zt1; x = zt1 + zt2; \
h=g; g=f; f=e; e=d; d=c; c=b; b=a; a=x; }
#define ZP_WMIX(w) { \
for (int zi = 0; zi < 16; zi++) w[zi] += s1(w[(zi+14)&15]) + w[(zi+9)&15] + s0(w[(zi+1)&15]); }
__device__ __forceinline__ void zlab_sha256_pair_h0(uint32_t *w0, uint32_t *w1, uint32_t *out0, uint32_t *out1) {
uint32_t a0=I[0],b0=I[1],c0=I[2],d0=I[3],e0=I[4],f0=I[5],g0=I[6],h0=I[7],x0;
uint32_t a1=I[0],b1=I[1],c1=I[2],d1=I[3],e1=I[4],f1=I[5],g1=I[6],h1=I[7],x1;
#pragma unroll 1
for (int blk = 0; blk < 64; blk += 16) {
if (blk) { ZP_WMIX(w0); ZP_WMIX(w1); }
ZP_RND2(blk);
}
out0[0]=I[0]+a0;out0[1]=I[1]+b0;out0[2]=I[2]+c0;out0[3]=I[3]+d0;
out0[4]=I[4]+e0;out0[5]=I[5]+f0;out0[6]=I[6]+g0;out0[7]=I[7]+h0;
out1[0]=I[0]+a1;out1[1]=I[1]+b1;out1[2]=I[2]+c1;out1[3]=I[3]+d1;
out1[4]=I[4]+e1;out1[5]=I[5]+f1;out1[6]=I[6]+g1;out1[7]=I[7]+h1;
}
#endif
__device__ __forceinline__ int gpu_bench_valid_words(const uint32_t *hs) {
int ok = 1;
#pragma unroll
for (int i = 0; i < QSB_ZEROS_N / 32; i++) ok &= (hs[i] == 0u);
#if (QSB_ZEROS_N % 32) != 0
ok &= ((hs[QSB_ZEROS_N / 32] >> (32 - (QSB_ZEROS_N % 32))) == 0u);
#endif
return ok;
}






#define MAX_N 150
#define MAX_T 16
#define SIG_PUSH_SIZE 10









#define QSB_FAST_N_INC 30
#define QSB_FAST_N_CONST 69
#define QSB_PREFIX_BLOCKS 2
#include "prefix_cache.cuh"









#define QSB_SE_N_INC 10
#define QSB_SE_EARLY 6
#define QSB_SE_TWIN 3
#define QSB_SE_CUT 137








#ifndef QSB_SE_WINDOWS
#define QSB_SE_WINDOWS 128
#endif
#ifndef QSB_SE_BLOCK
#define QSB_SE_BLOCK 256
#endif
#ifndef QSB_CTA_WINDOW_SPLIT
#define QSB_CTA_WINDOW_SPLIT 0
#endif
#if QSB_CTA_WINDOW_SPLIT
#if QSB_SE_BLOCK != 64 || QSB_SE_WINDOWS != 128 || QSB_ROOT_LUT_SMEM
#error "QSB_CTA_WINDOW_SPLIT requires 64-thread CTAs, 128 windows and no root shared LUT"
#endif
#define QSB_CTA_PARTS 2
#define QSB_SE_HALVES 1
#else
#if (QSB_SE_BLOCK != 128 && QSB_SE_BLOCK != 256) || QSB_SE_BLOCK < QSB_SE_WINDOWS || QSB_SE_BLOCK % QSB_SE_WINDOWS
#error "QSB_SE_BLOCK must be 128 or 256 and contain complete window sets"
#endif
#define QSB_CTA_PARTS 1
#define QSB_SE_HALVES (QSB_SE_BLOCK / QSB_SE_WINDOWS)
#endif
#define QSB_SE_PER_EPOCH QSB_SE_WINDOWS

#ifndef ZLAB_LAUNCH_BLOCKS
#define ZLAB_LAUNCH_BLOCKS 262144
#endif
#define QSB_SE_LAUNCH_BLOCKS ZLAB_LAUNCH_BLOCKS






typedef struct {
uint32_t mid[8];
uint32_t remW[2];
uint8_t early[QSB_SE_EARLY];
uint8_t pad[64 - 8 * 4 - 2 * 4 - QSB_SE_EARLY];
} epoch_desc_t;
static_assert(sizeof(epoch_desc_t) == 64, "epoch_desc_t must stay 64 bytes");






__device__ __constant__ uint8_t WIN3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
#include "window_schedule_shared.cuh"





__device__ __forceinline__ void unrank_combo(uint64_t rank, int n, int t, uint8_t *out) {
int lo = 0;
for (int i = 0; i < t; i++) {
int k = t - i - 1;
int hi = n - (t - i);

while (lo < hi) {
int mid = (lo + hi) >> 1;

uint64_t a = BINOM_C[n - lo][k + 1];
uint64_t b = BINOM_C[n - mid - 1][k + 1];
uint64_t pref = (a >= b) ? (a - b) : 0;
if (rank < pref) hi = mid;
else { rank -= pref; lo = mid + 1; }
}
out[i] = (uint8_t)lo;
lo++;
}
}








#if !QSB_TRIM_DIRECT_PRODUCER
__global__ void kernel_build_epochs(
uint64_t epoch_base, uint64_t n_epochs,
int window_start, int s_early,
const uint32_t * __restrict__ d_midstate,
const uint8_t * __restrict__ d_prefix_remainder, int prefix_remainder_len,
const uint8_t * __restrict__ d_dummy_sigs,
epoch_desc_t * __restrict__ d_epochs
#if ZLAB_HITPATH
, uint32_t *d_hit_reset
#endif
)
{
int t = blockIdx.x * blockDim.x + threadIdx.x;
#if ZLAB_HITPATH

if (t == 0) *d_hit_reset = 0;
#endif
uint64_t e = epoch_base + (uint64_t)t;
if (e >= n_epochs) return;
uint8_t early[MAX_T];
unrank_combo(e, window_start, s_early, early);
uint32_t state[8];
for (int i = 0; i < 8; i++) state[i] = d_midstate[i];
uint32_t curW[16];
uint8_t *cur = (uint8_t *)curW;
int cur_pos = 0;
for (int i = 0; i < prefix_remainder_len; i++) {
cur[cur_pos++] = d_prefix_remainder[i];
if (cur_pos == 64) {
uint32_t blk[16];
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}
int sel = 0;
for (int i = 0; i < window_start; i++) {
if (sel < s_early && (int)early[sel] == i) { sel++; continue; }
const uint8_t *row = d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE;
for (int b = 0; b < SIG_PUSH_SIZE; b++) {
cur[cur_pos++] = row[b];
if (cur_pos == 64) {
uint32_t blk[16];
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}
}
epoch_desc_t *d = d_epochs + t;
for (int i = 0; i < 8; i++) d->mid[i] = state[i];


d->remW[0] = bswap32(curW[0]);
d->remW[1] = bswap32(curW[1]);
for (int i = 0; i < s_early; i++) d->early[i] = early[i];
}
#endif
#ifndef QSB_EPOCH_GROUPS
#define QSB_EPOCH_GROUPS 1
#endif
#if QSB_EPOCH_GROUPS
#include "epoch_groups.cuh"
#endif






__device__ __forceinline__ void qsb_field_mul_raw(uint64_t *out,uint64_t *a,uint64_t *b){
uint64_t r0,r1,r2,r3;
asm(
"{\n"
"\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
"\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
"\t.reg .u32 cy,o15;\n"
"\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
"\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
"\tmov.b64 {a0,a1}, %4;\n"
"\tmov.b64 {a2,a3}, %5;\n"
"\tmov.b64 {a4,a5}, %6;\n"
"\tmov.b64 {a6,a7}, %7;\n"
"\tmov.b64 {b0,b1}, %8;\n"
"\tmov.b64 {b2,b3}, %9;\n"
"\tmov.b64 {b4,b5}, %10;\n"
"\tmov.b64 {b6,b7}, %11;\n"
"\tmul.wide.u32 e0, a0, b0; mul.wide.u32 e1, a0, b2; mul.wide.u32 e2, a0, b4; mul.wide.u32 e3, a0, b6;\n"
"\tmul.wide.u32 t, a1, b1; add.cc.u64 e1, e1, t;\n"
"\tmul.wide.u32 t, a1, b3; addc.cc.u64 e2, e2, t;\n"
"\tmul.wide.u32 t, a1, b5; addc.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a1, b7; addc.u64 e4, t, 0;\n"
"\tmul.wide.u32 t, a2, b0; add.cc.u64 e1, e1, t;\n"
"\tmul.wide.u32 t, a2, b2; addc.cc.u64 e2, e2, t;\n"
"\tmul.wide.u32 t, a2, b4; addc.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a2, b6; addc.cc.u64 e4, e4, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a3, b1; add.cc.u64 e2, e2, t;\n"
"\tmul.wide.u32 t, a3, b3; addc.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a3, b5; addc.cc.u64 e4, e4, t;\n"
"\tmul.wide.u32 t, a3, b7; addc.u64 e5, t, lc;\n"
"\tmul.wide.u32 t, a4, b0; add.cc.u64 e2, e2, t;\n"
"\tmul.wide.u32 t, a4, b2; addc.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a4, b4; addc.cc.u64 e4, e4, t;\n"
"\tmul.wide.u32 t, a4, b6; addc.cc.u64 e5, e5, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a5, b1; add.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a5, b3; addc.cc.u64 e4, e4, t;\n"
"\tmul.wide.u32 t, a5, b5; addc.cc.u64 e5, e5, t;\n"
"\tmul.wide.u32 t, a5, b7; addc.u64 e6, t, lc;\n"
"\tmul.wide.u32 t, a6, b0; add.cc.u64 e3, e3, t;\n"
"\tmul.wide.u32 t, a6, b2; addc.cc.u64 e4, e4, t;\n"
"\tmul.wide.u32 t, a6, b4; addc.cc.u64 e5, e5, t;\n"
"\tmul.wide.u32 t, a6, b6; addc.cc.u64 e6, e6, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a7, b1; add.cc.u64 e4, e4, t;\n"
"\tmul.wide.u32 t, a7, b3; addc.cc.u64 e5, e5, t;\n"
"\tmul.wide.u32 t, a7, b5; addc.cc.u64 e6, e6, t;\n"
"\tmul.wide.u32 t, a7, b7; addc.u64 e7, t, lc;\n"
"\tmul.wide.u32 o0, a0, b1; mul.wide.u32 o1, a0, b3; mul.wide.u32 o2, a0, b5; mul.wide.u32 o3, a0, b7;\n"
"\tmul.wide.u32 t, a1, b0; add.cc.u64 o0, o0, t;\n"
"\tmul.wide.u32 t, a1, b2; addc.cc.u64 o1, o1, t;\n"
"\tmul.wide.u32 t, a1, b4; addc.cc.u64 o2, o2, t;\n"
"\tmul.wide.u32 t, a1, b6; addc.cc.u64 o3, o3, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a2, b1; add.cc.u64 o1, o1, t;\n"
"\tmul.wide.u32 t, a2, b3; addc.cc.u64 o2, o2, t;\n"
"\tmul.wide.u32 t, a2, b5; addc.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a2, b7; addc.u64 o4, t, lc;\n"
"\tmul.wide.u32 t, a3, b0; add.cc.u64 o1, o1, t;\n"
"\tmul.wide.u32 t, a3, b2; addc.cc.u64 o2, o2, t;\n"
"\tmul.wide.u32 t, a3, b4; addc.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a3, b6; addc.cc.u64 o4, o4, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a4, b1; add.cc.u64 o2, o2, t;\n"
"\tmul.wide.u32 t, a4, b3; addc.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a4, b5; addc.cc.u64 o4, o4, t;\n"
"\tmul.wide.u32 t, a4, b7; addc.u64 o5, t, lc;\n"
"\tmul.wide.u32 t, a5, b0; add.cc.u64 o2, o2, t;\n"
"\tmul.wide.u32 t, a5, b2; addc.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a5, b4; addc.cc.u64 o4, o4, t;\n"
"\tmul.wide.u32 t, a5, b6; addc.cc.u64 o5, o5, t;\n"
"\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
"\tmul.wide.u32 t, a6, b1; add.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a6, b3; addc.cc.u64 o4, o4, t;\n"
"\tmul.wide.u32 t, a6, b5; addc.cc.u64 o5, o5, t;\n"
"\tmul.wide.u32 t, a6, b7; addc.u64 o6, t, lc;\n"
"\tmul.wide.u32 t, a7, b0; add.cc.u64 o3, o3, t;\n"
"\tmul.wide.u32 t, a7, b2; addc.cc.u64 o4, o4, t;\n"
"\tmul.wide.u32 t, a7, b4; addc.cc.u64 o5, o5, t;\n"
"\tmul.wide.u32 t, a7, b6; addc.cc.u64 o6, o6, t;\n"
"\taddc.u32 o15, 0, 0;\n"
"\tmov.b64 {x0,x1}, e0;\n"
"\tmov.b64 {x2,x3}, e1;\n"
"\tmov.b64 {x4,x5}, e2;\n"
"\tmov.b64 {x6,x7}, e3;\n"
"\tmov.b64 {x8,x9}, e4;\n"
"\tmov.b64 {x10,x11}, e5;\n"
"\tmov.b64 {x12,x13}, e6;\n"
"\tmov.b64 {x14,x15}, e7;\n"
"\tmov.b64 {y1,y2}, o0;\n"
"\tmov.b64 {y3,y4}, o1;\n"
"\tmov.b64 {y5,y6}, o2;\n"
"\tmov.b64 {y7,y8}, o3;\n"
"\tmov.b64 {y9,y10}, o4;\n"
"\tmov.b64 {y11,y12}, o5;\n"
"\tmov.b64 {y13,y14}, o6;\n"
"\tadd.cc.u32 x1, x1, y1;\n"
"\taddc.cc.u32 x2, x2, y2;\n"
"\taddc.cc.u32 x3, x3, y3;\n"
"\taddc.cc.u32 x4, x4, y4;\n"
"\taddc.cc.u32 x5, x5, y5;\n"
"\taddc.cc.u32 x6, x6, y6;\n"
"\taddc.cc.u32 x7, x7, y7;\n"
"\taddc.cc.u32 x8, x8, y8;\n"
"\taddc.cc.u32 x9, x9, y9;\n"
"\taddc.cc.u32 x10, x10, y10;\n"
"\taddc.cc.u32 x11, x11, y11;\n"
"\taddc.cc.u32 x12, x12, y12;\n"
"\taddc.cc.u32 x13, x13, y13;\n"
"\taddc.cc.u32 x14, x14, y14;\n"
"\taddc.u32 x15, x15, o15;\n"
"\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n"
"\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n"
"\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n"
"\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n"
"\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n"
"\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n"
"\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n"
"\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n"
"\taddc.u32 f8, 0, 0;\n"
"\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
"\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
"\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
"\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
"\taddc.u32 g8, 0, 0;\n"
"\tmov.b64 {z0,z1}, f0;\n"
"\tmov.b64 {z2,z3}, f1;\n"
"\tmov.b64 {z4,z5}, f2;\n"
"\tmov.b64 {z6,z7}, f3;\n"
"\tmov.b64 {w0,w1}, g0;\n"
"\tmov.b64 {w2,w3}, g1;\n"
"\tmov.b64 {w4,w5}, g2;\n"
"\tmov.b64 {w6,w7}, g3;\n"
"\tadd.cc.u32  z1, z1, w0;\n"
"\taddc.cc.u32 z2, z2, w1;\n"
"\taddc.cc.u32 z3, z3, w2;\n"
"\taddc.cc.u32 z4, z4, w3;\n"
"\taddc.cc.u32 z5, z5, w4;\n"
"\taddc.cc.u32 z6, z6, w5;\n"
"\taddc.cc.u32 z7, z7, w6;\n"
"\taddc.cc.u32 z8, f8, w7;\n"
"\taddc.u32    z9, g8, 0;\n"
"\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n"
"\tmad.lo.u32 m1, z9, 977, m1;\n"
"\tadd.cc.u32 m1, m1, z8;\n"
"\taddc.u32 m2, z9, 0;\n"
"\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n"
"\taddc.cc.u32 z3, z3, 0;\n"
"\taddc.cc.u32 z4, z4, 0;\n"
"\taddc.cc.u32 z5, z5, 0;\n"
"\taddc.cc.u32 z6, z6, 0;\n"
"\taddc.cc.u32 z7, z7, 0;\n"
"    .reg .u32 cf, k0, k1, v0, v1, v2, v3, v4, v5, v6, v7, borrow;\n"
"    .reg .pred take;\n"
"    addc.u32 cf, 0, 0;\n"
"    mul.lo.u32 k0, cf, 977;\n"
"    add.cc.u32 z0, z0, k0;\n"
"    addc.cc.u32 z1, z1, cf;\n"
"    addc.cc.u32 z2, z2, 0;\n"
"    addc.cc.u32 z3, z3, 0;\n"
"    addc.cc.u32 z4, z4, 0;\n"
"    addc.cc.u32 z5, z5, 0;\n"
"    addc.cc.u32 z6, z6, 0;\n"
"    addc.u32 z7, z7, 0;\n"
"mov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
"\t}\n"
: "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
: "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=0;
}

__device__ __forceinline__ void qsb_field_normalize(uint64_t *r){
if ((r[1] & r[2] & r[3]) == UINT64_MAX && r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
r[0] -= 0xFFFFFFFEFFFFFC2FULL;
r[1] = r[2] = r[3] = 0;
}
}
__device__ __forceinline__ void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b){
qsb_field_mul_raw(out,a,b);
qsb_field_normalize(out);
}





















__device__ __forceinline__ void qsb_affine_finish_prepare(uint64_t *X, uint64_t *Z, uint64_t *xR, uint64_t *D, uint64_t *W) {
uint64_t t[4];
_ModMult(t, xR, Z);
_ModSub256(D, t, X);
W[4] = 0;
_ModMult(W, Z, D);
}


__device__ __forceinline__ void qsb_affine_finish(uint64_t *X, uint64_t *Y, uint64_t *Z, uint64_t *D, uint64_t *inv,
uint64_t *xR, uint64_t *yR,
uint64_t *x1, uint64_t *y1, uint64_t *x2, uint64_t *y2) {
uint64_t iZ[4], xP[4], yP[4], z2[4], id[4], s[4], t[4], m1[4], m2[4], sq[4], xs[4];
_ModMult(iZ, inv, D);
_ModMult(xP, X, iZ);
_ModMult(yP, Y, iZ);
_ModSqr(z2, Z);
_ModMult(id, inv, z2);
_ModSub256(s, yR, yP);
_ModMult(m1, s, id);
_ModAdd256(t, yR, yP);
_ModMult(m2, t, id);
_ModAdd256(xs, xP, xR);
_ModSqr(sq, m1);
_ModSub256(x1, sq, xs);
_ModSub256(t, xP, x1);
_ModMult(y1, m1, t);
_ModSub256(y1, y1, yP);
_ModSqr(sq, m2);
_ModSub256(x2, sq, xs);
_ModSub256(t, xP, x2);
_ModMult(y2, m2, t);
_ModAdd256(y2, y2, yP);
_ModNeg256(y2);
}





__device__ __forceinline__ void qsb_xyzz_finish_prepare(
uint64_t *X_D, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *xR, uint64_t *W
) {
uint64_t t[4];
#ifndef QSB_ISO_FAST_X
#define QSB_ISO_FAST_X 1
#endif
#if QSB_ISO_FAST_X
(void)xR;
if (QSB_ISO_XNEG) {
uint64_t zero[4]={0,0,0,0};
_ModSub256(t,zero,ZZ);
} else {
Load256(t,ZZ);
}
#else
_ModMult(t,xR,ZZ);
#endif
_ModSub256(t, t, X_D);
Load256(X_D, t);
_ModMult(W, ZZZ, X_D);
W[4] = 0;
}













__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(
uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
uint64_t *inv, uint64_t *xR, uint64_t *yR,
uint64_t *x1, uint64_t *x2
) {
uint64_t yb[4], m1[4], m2[4], t[4], s[4];
uint64_t cc[4]={QSB_U2R_C[0],QSB_U2R_C[1],QSB_U2R_C[2],QSB_U2R_C[3]};

_ModMult(yb, yR, ZZZ);
_ModMult(ZZ, inv);

_ModSub256(m1, yb, Y);
_ModMult(m1, ZZ);
_ModAdd256(m2, yb, Y);
_ModMult(m2, ZZ);
_ModAdd256(s, m1, m2);

_ModSub256(t, m1, cc);
_ModMult(x1, s, t);
_ModAdd256(x1, x1, xR);
_ModSub256(t, xR, x1);
_ModMult(t, m1);
_ModSub256(t, yR);
uint32_t parities = (uint32_t)(t[0] & 1ULL);

_ModSub256(t, m2, cc);
_ModMult(x2, s, t);
_ModAdd256(x2, x2, xR);
_ModSub256(t, xR, x2);
_ModMult(t, m2);
_ModSub256(t, yR);

parities |= (uint32_t)(((t[0] & 1ULL) ^ 1ULL) << 1);
return parities;
}

#include "tree_inverse.cuh"
#include "pair_shared.cuh"
#if QSB_TREE_STATIC_N && (!QSB_PAIR_SHARED || !ZLAB_TRIM)
#error "static tree extent requires the paired ranked-only CTA launch contract"
#endif
#if QSB_TREE_CONST_ADDR && (!QSB_PAIR_SHARED || !ZLAB_TRIM)
#error "constant tree addresses require the paired ranked-only CTA launch contract"
#endif
#if QSB_TREE_LEAF_SHFL && !(QSB_PAIR_SHARED && QSB_SC_PARK && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA && QSB_PARK128)
#error "leaf shuffle requires production paired parked fronts"
#endif
#if QSB_TREE_BOTTOM_SHFL && !QSB_PAIR_SHARED
#error "bottom shuffle requires paired candidate ownership"
#endif
#if QSB_FRONT3_PUBLISH_AB && !(QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA && QSB_PARK128 && QSB_SC_PARK && ZLAB_TREE == 2 && !QSB_TREE_ROW128 && !QSB_ROOT_LUT_SMEM && !QSB_ROOT_PARK_B && !QSB_PRE3_ROOT && !QSB_FRONT3_PUBLISH_A && !QSB_FRONT3_WORDS_ONLY && !QSB_OUTER_FRONT_CALL)
#error "Common publication requires production packed parking and reusable inverse arena"
#endif
#if QSB_FRONT3_PUBLISH_A && !(QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA && QSB_PARK128 && !QSB_FRONT3_WORDS_ONLY && !QSB_OUTER_FRONT_CALL)
#error "A publication requires the production paired packed-park front ABI"
#endif
#if QSB_FRONT3_WORDS_ONLY && !(QSB_PAIR_SHARED && ZLAB_K2S3M)
#error "Words-only front ABI requires the paired K2S3M path"
#endif
#if QSB_OUTER_FRONT_CALL && !(QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA && !QSB_OUTER_PAIR)
#error "Outer-front relocation requires paired dual-epoch SHA, K2S3M, and no OUTER_PAIR"
#endif
#if (QSB_GATE_REC_ROLL || QSB_GATE_REC_CALL) && !(QSB_GATE_H0 && QSB_GATE_H0_FMA && defined(QSB_ZEROS_N) && QSB_ZEROS_N >= 1 && QSB_ZEROS_N <= 32)
#error "Recovery gate experiments require FMA H0 with QSB_ZEROS_N in 1..32"
#endif
#if QSB_YNEG_FOLD && !ZLAB_K2S3M
#error "QSB_YNEG_FOLD is written for the ZLAB_K2S3M front (qsb_k2s_pre3 is the reader of the chain's Y)"
#endif
#if QSB_PRE3_ROOT
#if !(QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA)
#error "QSB_PRE3_ROOT is written for the paired digest path with qsb_pair_front3_z_value (QSB_PAIR_SHARED, ZLAB_K2S3M, ZLAB_DUAL_EPOCH_SHA)"
#endif





struct QsbPre3Idle{
uint64_t *nB;uint64_t (*parkA)[QSB_SE_BLOCK];int tid;
__device__ __forceinline__ void operator()()const{
uint64_t yR[4]={QSB_U2R_ISO[4],QSB_U2R_ISO[5],QSB_U2R_ISO[6],QSB_U2R_ISO[7]};
uint64_t y[4],zz[4],zzz[4],w[12];
#pragma unroll
for(int k=0;k<4;k++){y[k]=parkA[k][tid];zzz[k]=parkA[4+k][tid];zz[k]=0;}
qsb_k2s_pre3(y,zz,zzz,yR,w);
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=w[k];
#pragma unroll
for(int k=0;k<4;k++){y[k]=nB[k];zzz[k]=nB[4+k];zz[k]=nB[8+k];}
qsb_k2s_pre3(y,zz,zzz,yR,w);
#pragma unroll
for(int k=0;k<8;k++)nB[k]=w[k];
}
};
#endif

#if QSB_DEN_CROSS_IDLE
struct QsbDenCrossIdle{
uint64_t *nB,*prodA,*prodB;ulonglong2 (*parkA2)[QSB_SE_BLOCK];int tid;
__device__ __forceinline__ void operator()()const{
uint64_t zz[5],scaled[5];
#pragma unroll
for(int j=0;j<2;j++){
const ulonglong2 v=parkA2[4+j][tid];zz[2*j]=v.x;zz[2*j+1]=v.y;
}
zz[4]=0;QSB_TREE_MUL(scaled,zz,prodB);
#pragma unroll
for(int j=0;j<2;j++)parkA2[4+j][tid]=make_ulonglong2(scaled[2*j],scaled[2*j+1]);
#pragma unroll
for(int j=0;j<4;j++)zz[j]=nB[8+j];
zz[4]=0;QSB_TREE_MUL(scaled,zz,prodA);
#pragma unroll
for(int j=0;j<4;j++)nB[8+j]=scaled[j];
}
};
#endif

#if !QSB_HOST_VERIFY


__global__ void kernel_verify_pair_hits(
const uint8_t*tentative,uint8_t*verified,const epoch_desc_t*epochs,
const uint32_t*first,const uint8_t*gtable,int epochs_in_batch){
if(threadIdx.x==0)*((uint32_t*)verified)=0;
__syncthreads();
const uint32_t count=*((const uint32_t*)tentative);
const uint32_t limit=count<1024u?count:1024u;
for(uint32_t i=threadIdx.x;i<limit;i+=blockDim.x){
const uint8_t*record=tentative+4+(size_t)i*ZLAB_HIT_REC;
const uint32_t index=*((const uint32_t*)record)&0x3fffffffu;

const uint32_t ep=index/(uint32_t)QSB_SE_WINDOWS,lane=index&(uint32_t)(QSB_SE_WINDOWS-1);
if(ep>=(uint32_t)epochs_in_batch)continue;
const int encoded=qsb_pair_verify_candidate(
epochs+ep,first+(size_t)ep*QSB_FIRST_SLOTS*8,lane,gtable);
if(!encoded)continue;
const uint32_t slot=atomicAdd((uint32_t*)verified,1u);
if(slot<1024u){
uint8_t*out=verified+4+(size_t)slot*ZLAB_HIT_REC;
*((uint32_t*)out)=index|((uint32_t)(encoded-1)<<30);


for(int j=0;j<6;j++)out[4+j]=epochs[ep].early[j];
for(int j=0;j<3;j++)out[10+j]=WIN3[lane][j];
}
}
}
#endif



#if QSB_FIRST_HOSTLESS && !QSB_FIRST_DEVICE
#error "Descriptor-only host ring requires first-device prepass"
#endif
#if QSB_FIRST_DEVICE && (QSB_FIRST_LOCAL || !QSB_SLOT_PIPELINE || !QSB_HOST_PRODUCERS || !QSB_HOST_VERIFY || !QSB_FIRST_PACK8 || QSB_FIRST_PLANES || QSB_HP_FIRST_ONLY_UPLOAD)
#error "First-device prepass requires original packed verified host-v3 slot path"
#endif
#if QSB_FIRST_LOCAL && !(QSB_PAIR_SHARED && ZLAB_DUAL_EPOCH_SHA && ZLAB_K2S3M && QSB_SE_BLOCK==128 && QSB_SE_WINDOWS==128 && QSB_PARK128 && QSB_SLOT_PIPELINE && QSB_HOST_PRODUCERS && QSB_HOST_VERIFY && QSB_FIRST_PACK8 && !QSB_FIRST_PLANES && !QSB_HP_FIRST_ONLY_UPLOAD)
#error "Consumer-local first requires packed paired128 PARK128 host-v3 verified slot path"
#endif
#if QSB_DIGEST_PHASE_AUDIT && !(QSB_PAIR_SHARED && ZLAB_DUAL_EPOCH_SHA && ZLAB_K2S3M && QSB_SE_BLOCK==128 && QSB_SLOT_PIPELINE)
#error "Digest phase audit requires paired 128-thread slot pipeline"
#endif
__global__ void __launch_bounds__(QSB_SE_BLOCK, 512 / QSB_SE_BLOCK) kernel_digest(
const uint8_t * __restrict__ d_combos,
int n_pool, int t_sel,
const uint32_t * __restrict__ d_midstate,
const uint8_t * __restrict__ d_prefix_remainder,
int prefix_remainder_len,
const uint8_t * __restrict__ d_dummy_sigs,
const uint8_t * __restrict__ d_tail,
int tail_len,
const uint8_t * __restrict__ d_tx_suffix,
int tx_suffix_len,
int total_preimage_len,
const uint64_t * __restrict__ d_nri,
const uint64_t * __restrict__ d_u2rx, const uint64_t * __restrict__ d_u2ry,
const uint64_t * __restrict__ d_neg2u2rx, const uint64_t * __restrict__ d_neg2u2ry,
uint8_t * __restrict__ d_gt,
uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
uint8_t *d_hit_combos, uint8_t *d_hit_sighash,
uint8_t *d_hit_keynonce, uint8_t *d_hit_pubhash,
uint8_t *d_hit_qx, uint8_t *d_hit_qy,
int batch_size, int easy_mode, int single_hash, int calibrate_mode,
int window_start, uint64_t enum_base,
int t_win, int s_early, const uint8_t * __restrict__ d_early,
int fast_inc, const uint32_t * __restrict__ d_const_words,
const epoch_desc_t * __restrict__ d_epochs
, const uint32_t *d_first, int epochs_in_batch
) {
QSB_DP_STAMP(0);
#if QSB_PAIR_SHARED
const int tid = threadIdx.x;
#if QSB_TREE_BOTTOM_SHFL

const int candidate_tid=(tid>>1)|((tid&1)*(QSB_SE_BLOCK>>1));
#else
const int candidate_tid=tid;
#endif
#if QSB_CTA_WINDOW_SPLIT
const int lane = tid + (blockIdx.x & 1u) * QSB_SE_BLOCK;
#else
const int lane = candidate_tid & (QSB_SE_WINDOWS-1);
#endif
#if QSB_TID_UNSIGNED
const int half = (int)((unsigned)candidate_tid / (unsigned)QSB_SE_WINDOWS);
#else
const int half = candidate_tid / QSB_SE_WINDOWS;
#endif
int idx = blockIdx.x * blockDim.x + candidate_tid;
if(blockIdx.x*blockDim.x>=batch_size)return;
#if QSB_ROOT_LUT_SMEM
qsb_root_lut_issue(tid,blockDim.x);
#endif
const unsigned eA = (unsigned)QSB_PAIR_MUL*(blockIdx.x / QSB_CTA_PARTS) + 2u*(unsigned)half;
const bool hasA = eA < (unsigned)epochs_in_batch;
const bool active = idx<batch_size && hasA;
#if ZLAB_K2S3M && QSB_PARK128
#if !ZLAB_DUAL_EPOCH_SHA || QSB_TAIL_STAGGER || QSB_TAIL_WEAVE || QSB_PRE3_ROOT
#error "QSB_PARK128 is written for the paired-SHA front with N-b2's tails (no stagger, weave or pre3 in the root)"
#endif
__shared__ __align__(16) ulonglong2 parkA2[6][QSB_SE_BLOCK];
#elif ZLAB_K2S3M
__shared__ uint64_t parkA[12][QSB_SE_BLOCK];
#if QSB_TAIL_WEAVE
static_assert(sizeof(parkA[0])==QSB_WEAVE_PARK_STRIDE,"QSB_WEAVE_PARK_STRIDE is the parkA row stride");
#endif
#else
__shared__ uint64_t parkA[8][QSB_SE_BLOCK];
#endif
const unsigned eA0 = hasA ? eA : 0u;
#if !(QSB_HIT_NO_COMBO && ZLAB_DUAL_EPOCH_SHA)

const epoch_desc_t *e0 = d_epochs + eA0;
#endif
const bool hasB = eA+1u < (unsigned)epochs_in_batch;
#if !(QSB_HIT_NO_COMBO && ZLAB_DUAL_EPOCH_SHA)
const epoch_desc_t *e1 = hasB ? e0+1 : e0;
#endif
#if QSB_FIRST_LOCAL
uint32_t *local_first=reinterpret_cast<uint32_t *>(&parkA2[0][0]);
if(tid<16) {
const unsigned member=(unsigned)tid>>3,cls=(unsigned)tid&7u;
const unsigned ep=member && hasB?eA0+1u:eA0;
qsb_local_first(d_epochs+ep,cls,local_first+member*64u);
}
__syncthreads();
const uint32_t *f0=local_first,*f1=local_first+64;
#else
const uint32_t *f0=d_first+(size_t)eA0*QSB_FIRST_SLOTS*8;
const uint32_t *f1=hasB?f0+QSB_FIRST_SLOTS*8:f0;
#endif
uint64_t u2rx[4]={QSB_U2R_ISO[0],QSB_U2R_ISO[1],QSB_U2R_ISO[2],QSB_U2R_ISO[3]};
uint64_t u2ry[4]={QSB_U2R_ISO[4],QSB_U2R_ISO[5],QSB_U2R_ISO[6],QSB_U2R_ISO[7]};
#if ZLAB_K2S3M
uint64_t prodA[5], prodB[5], nB[12];
#if ZLAB_DUAL_EPOCH_SHA
uint64_t zB[4];
#if QSB_OUTER_FRONT_CALL
uint32_t stateA[8],stateB[8];
qsb_scheduled_window_hash_pair(stateA,stateB,lane,f0,f1);
QsbPairEpochZ zpair;
#pragma unroll
for(int k=0;k<4;k++){
zpair.a[k]=((uint64_t)stateA[2*k+1]<<32)|stateA[2*k];
zpair.b[k]=((uint64_t)stateB[2*k+1]<<32)|stateB[2*k];
}
#else
QsbPairEpochZ zpair=qsb_pair_epoch_z_value(f0,f1,lane);
#endif
#if QSB_FIRST_LOCAL
/* All first-state readers retire before parking overwrites the same bytes. */
__syncthreads();
#endif
QSB_DP_STAMP(1);


#if QSB_PARK128
parkA2[4][tid]=make_ulonglong2(zpair.b[0],zpair.b[1]);
parkA2[5][tid]=make_ulonglong2(zpair.b[2],zpair.b[3]);
#else
#pragma unroll
for(int k=0;k<4;k++)parkA[8+k][tid]=zpair.b[k];
#endif
#endif
#else
uint64_t prodA[5], prodB[5], m1B[4], m2B[4];
#endif
int okA, okB;
{
#if ZLAB_K2S3M
#if ZLAB_DUAL_EPOCH_SHA
#if QSB_FRONT3_PUBLISH_AB
#if QSB_FRONT3_PUBLISH_AB == 2
QsbPairFront3PublishedAB fa=qsb_pair_front3_publish_ab(zpair.a[0],zpair.a[1],zpair.a[2],zpair.a[3],d_gt,
(uint32_t)__cvta_generic_to_shared(&parkA2[0][tid]) QSB_R_PASS(u2rx,u2ry));
#else
QsbPairFront3PublishedAB fa=qsb_pair_front3_publish_ab(zpair.a[0],zpair.a[1],zpair.a[2],zpair.a[3],d_gt,
(uint32_t)__cvta_generic_to_shared(&parkA2[0][tid]),
(uint32_t)__cvta_generic_to_shared(&parkA2[1][tid]),
(uint32_t)__cvta_generic_to_shared(&parkA2[2][tid]),
(uint32_t)__cvta_generic_to_shared(&parkA2[3][tid]),8u,8u QSB_R_PASS(u2rx,u2ry));
#endif
#elif QSB_FRONT3_PUBLISH_A
QsbPairFront3PublishedA fa=qsb_pair_front3_publish_a(zpair.a[0],zpair.a[1],zpair.a[2],zpair.a[3],d_gt,
(uint32_t)__cvta_generic_to_shared(&parkA2[0][tid]) QSB_R_PASS(u2rx,u2ry));
#else
QsbPairFront3 fa=qsb_pair_front3_z_value(zpair.a[0],zpair.a[1],zpair.a[2],zpair.a[3],d_gt QSB_R_PASS(u2rx,u2ry));
#endif
#else
QsbPairFront3 fa=qsb_pair_front3_value(e0,f0,lane,d_gt,u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
#endif
Load256(prodA,fa.words);prodA[4]=0;
#if QSB_FRONT3_WORDS_ONLY
const int raw_okA=(prodA[0]|prodA[1]|prodA[2]|prodA[3])!=0;
#else
const int raw_okA=fa.ok;
#endif
okA=raw_okA && active;
#if QSB_OK_FOLD == 2
prodA[0]|=(uint64_t)(!okA);
#elif QSB_OK_FOLD == 1
prodA[0]|=(uint64_t)(raw_okA==0);
#else
if(!okA){prodA[0]=1;prodA[1]=prodA[2]=prodA[3]=prodA[4]=0;}
#endif
#if QSB_SC_PARK


#if QSB_TREE_ROW128
QTR_ST4(qsb_sc_products2,tid,prodA);
#else
#pragma unroll
for(int k=0;k<4;k++)qsb_sc_products[k][candidate_tid
#if QSB_TREE_LEAF_SHFL
+ QSB_SE_BLOCK
#endif
]=prodA[k];
#endif
#endif
#if ZLAB_DUAL_EPOCH_SHA && QSB_PARK128
#if !QSB_FRONT3_PUBLISH_A && !QSB_FRONT3_PUBLISH_AB
#pragma unroll
for(int j=0;j<4;j++)parkA2[j][tid]=make_ulonglong2(fa.words[4+2*j],fa.words[5+2*j]);
#endif
{ const ulonglong2 b01=parkA2[4][tid],b23=parkA2[5][tid]; zB[0]=b01.x;zB[1]=b01.y;zB[2]=b23.x;zB[3]=b23.y; }
#if QSB_FRONT3_PUBLISH_A || QSB_FRONT3_PUBLISH_AB
parkA2[4][tid]=make_ulonglong2(fa.words[4],fa.words[5]);
parkA2[5][tid]=make_ulonglong2(fa.words[6],fa.words[7]);
#else
parkA2[4][tid]=make_ulonglong2(fa.words[12],fa.words[13]);
parkA2[5][tid]=make_ulonglong2(fa.words[14],fa.words[15]);
#endif
#elif ZLAB_DUAL_EPOCH_SHA
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=fa.words[4+k];
#pragma unroll
for(int k=0;k<4;k++)zB[k]=parkA[8+k][tid];
#pragma unroll
for(int k=0;k<4;k++)parkA[8+k][tid]=fa.words[12+k];
#else
#pragma unroll
for(int k=0;k<12;k++)parkA[k][tid]=fa.words[4+k];
#endif
#else
uint64_t m1[4],m2[4];
QsbPairFront fa=qsb_pair_front_value(e0,f0,tid,d_gt,u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
Load256(prodA,fa.words);prodA[4]=0;Load256(m1,fa.words+4);Load256(m2,fa.words+8);
okA=fa.ok && active;
if(!okA){prodA[0]=1;prodA[1]=prodA[2]=prodA[3]=prodA[4]=0;}
#pragma unroll
for(int k=0;k<4;k++){parkA[k][tid]=m1[k];parkA[4+k][tid]=m2[k];}
#endif
}
QSB_DP_STAMP(2);

#if ZLAB_K2S3M
#if ZLAB_DUAL_EPOCH_SHA
#if QSB_FRONT3_PUBLISH_AB



#if QSB_FRONT3_PUBLISH_AB == 2
QsbPairFront3PublishedAB fb=qsb_pair_front3_publish_ab(zB[0],zB[1],zB[2],zB[3],d_gt,
1u QSB_R_PASS(u2rx,u2ry));
#else
QsbPairFront3PublishedAB fb=qsb_pair_front3_publish_ab(zB[0],zB[1],zB[2],zB[3],d_gt,
(uint32_t)__cvta_generic_to_shared(&qsb_sc_products[0][QSB_SE_BLOCK+tid]),
(uint32_t)__cvta_generic_to_shared(&qsb_sc_products[2][QSB_SE_BLOCK+tid]),
(uint32_t)__cvta_generic_to_shared(&qsb_front_inverse[0][tid]),
(uint32_t)__cvta_generic_to_shared(&qsb_front_inverse[2][tid]),
2u*QSB_SE_BLOCK*sizeof(uint64_t),QSB_SE_BLOCK*sizeof(uint64_t) QSB_R_PASS(u2rx,u2ry));
#endif
#else
QsbPairFront3 fb=qsb_pair_front3_z_value(zB[0],zB[1],zB[2],zB[3],d_gt QSB_R_PASS(u2rx,u2ry));
#endif
#else
QsbPairFront3 fb=qsb_pair_front3_value(e1,f1,lane,d_gt,u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
#endif
Load256(prodB,fb.words);prodB[4]=0;
#if QSB_FRONT3_PUBLISH_AB
#pragma unroll
for(int k=0;k<4;k++){
nB[k]=qsb_sc_products[k][QSB_SE_BLOCK+tid];
nB[4+k]=qsb_front_inverse[k][tid];
}
#pragma unroll
for(int k=0;k<4;k++)nB[8+k]=fb.words[4+k];
#else
#pragma unroll
for(int k=0;k<12;k++)nB[k]=fb.words[4+k];
#endif
#if QSB_PRE3_ROOT == 2


if(__all_sync(0xffffffffu,tid<32))QsbPre3Idle{nB,parkA,tid}();
#endif
#else
QsbPairFront fb=qsb_pair_front_value(e1,f1,tid,d_gt,u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
Load256(prodB,fb.words);prodB[4]=0;Load256(m1B,fb.words+4);Load256(m2B,fb.words+8);
#endif
#if QSB_FRONT3_WORDS_ONLY
const int raw_okB=(prodB[0]|prodB[1]|prodB[2]|prodB[3])!=0;
#else
const int raw_okB=fb.ok;
#endif
okB=raw_okB && active && hasB;
#if QSB_OK_FOLD == 2
prodB[0]|=(uint64_t)(!okB);
#elif QSB_OK_FOLD == 1
prodB[0]|=(uint64_t)(raw_okB==0);
#else
if(!okB){prodB[0]=1;prodB[1]=prodB[2]=prodB[3]=prodB[4]=0;}
#endif
#if QSB_SC_PARK
asm volatile("" ::: "memory");
#if QSB_TREE_ROW128
QTR_LD4(qsb_sc_products2,tid,prodA);
#else
#pragma unroll
for(int k=0;k<4;k++)prodA[k]=qsb_sc_products[k][candidate_tid
#if QSB_TREE_LEAF_SHFL
+ QSB_SE_BLOCK
#endif
];
#endif
#endif
QSB_DP_STAMP(3);
uint64_t leaf[5];
QSB_TREE_MUL(leaf,prodA,prodB);
#if QSB_DEN_CROSS_PRE
#if !ZLAB_K2S3M || !ZLAB_DUAL_EPOCH_SHA || !QSB_PARK128
#error "QSB_DEN_CROSS_PRE requires twelve-word paired fronts with vector parking"
#endif
#if QSB_DEN_CROSS_IDLE
QsbDenCrossIdle denIdle{nB,prodA,prodB,parkA2,tid};
if(__all_sync(0xffffffffu,tid<32))denIdle();
#else



{
uint64_t zz[5],scaled[5];
#if QSB_DEN_CROSS_PRE == 1 || QSB_DEN_CROSS_PRE == 3
#pragma unroll
for(int j=0;j<2;j++){
const ulonglong2 v=parkA2[4+j][tid];
zz[2*j]=v.x;zz[2*j+1]=v.y;
}
zz[4]=0;
QSB_TREE_MUL(scaled,zz,prodB);
#pragma unroll
for(int j=0;j<2;j++)parkA2[4+j][tid]=make_ulonglong2(scaled[2*j],scaled[2*j+1]);
#endif
#if QSB_DEN_CROSS_PRE != 3
#pragma unroll
for(int j=0;j<4;j++)zz[j]=nB[8+j];
zz[4]=0;
QSB_TREE_MUL(scaled,zz,prodA);
#pragma unroll
for(int j=0;j<4;j++)nB[8+j]=scaled[j];
#endif
}
#endif
#endif
QSB_DP_STAMP(4);
#if QSB_DEN_CROSS_IDLE
qsb_block_inverse_tree_x<0,QsbDenCrossIdle,0>(leaf,denIdle);
#elif QSB_ROOT_PARK_B
qsb_block_inverse_tree(leaf,nB);
#elif QSB_PRE3_ROOT
qsb_block_inverse_tree_x<QSB_ROOT_LUT_SMEM,QsbPre3Idle,QSB_ROOT_WARP>(leaf,QsbPre3Idle{nB,parkA,tid});
#elif QSB_ROOT_LUT_SMEM
qsb_block_inverse_tree_x<1,QsbTreeNoIdle,QSB_ROOT_WARP>(leaf,QsbTreeNoIdle());
#else
qsb_block_inverse_tree(leaf);
#endif
QSB_DP_STAMP(5);
#ifndef QSB_ISO_RELOAD_R
#define QSB_ISO_RELOAD_R 1
#endif
#if QSB_POOL_RCONST && !QSB_ISO_RELOAD_R
#error "QSB_POOL_RCONST reads original QSB_U2R and requires ISO_RELOAD_R"
#endif
#if QSB_R_CBANK && !QSB_ISO_RELOAD_R
#error "QSB_R_CBANK tail reads QSB_U2R; it needs QSB_ISO_RELOAD_R"
#endif
#if QSB_R_CBANK_TAILS && !QSB_ISO_RELOAD_R
#error "QSB_R_CBANK_TAILS: the tails read QSB_U2R from the constant bank; they need QSB_ISO_RELOAD_R"
#endif
#ifndef QSB_SC_LATE
#if QSB_PRE3_ROOT || QSB_TAIL_WEAVE
#define QSB_SC_LATE 1

#else
#define QSB_SC_LATE 0
#endif
#endif
#if QSB_SC_LATE



unsigned qsb_sc_bx,qsb_sc_tx;
asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(qsb_sc_bx));
asm volatile("mov.u32 %0, %%tid.x;" : "=r"(qsb_sc_tx));
const unsigned qsb_sc_eA=(unsigned)QSB_PAIR_MUL*(qsb_sc_bx / QSB_CTA_PARTS)+2u*(qsb_sc_tx/(unsigned)QSB_SE_WINDOWS);
const unsigned qsb_sc_eA0=qsb_sc_eA<(unsigned)epochs_in_batch?qsb_sc_eA:0u;
#if !QSB_HIT_NO_COMBO
const epoch_desc_t *qsb_sc_e0=d_epochs+qsb_sc_eA0;
const epoch_desc_t *qsb_sc_e1=(qsb_sc_eA+1u<(unsigned)epochs_in_batch)?qsb_sc_e0+1:qsb_sc_e0;
#endif
#define QSB_SC_EA0 qsb_sc_eA0
#define QSB_SC_E0 qsb_sc_e0
#define QSB_SC_E1 qsb_sc_e1
#else
#define QSB_SC_EA0 eA0
#define QSB_SC_E0 e0
#define QSB_SC_E1 e1
#endif
#if QSB_ISO_RELOAD_R && !QSB_R_CBANK_TAILS




u2rx[0]=QSB_U2R[0];u2rx[1]=QSB_U2R[1];u2rx[2]=QSB_U2R[2];u2rx[3]=QSB_U2R[3];
u2ry[0]=QSB_U2R[4];u2ry[1]=QSB_U2R[5];u2ry[2]=QSB_U2R[6];u2ry[3]=QSB_U2R[7];
#endif
#if ZLAB_K2S3M && QSB_TAIL_STAGGER





{
#if QSB_TAIL_STAGGER == 1
const bool qsb_ffgg = half != 0;
#elif QSB_TAIL_STAGGER == 2
const bool qsb_ffgg = false;
#elif QSB_TAIL_STAGGER == 3
const bool qsb_ffgg = true;
#elif QSB_TAIL_STAGGER == 4
unsigned qsb_wslot;
asm volatile("mov.u32 %0, %%warpid;" : "=r"(qsb_wslot));
const bool qsb_ffgg = ((qsb_wslot >> 2) & 1u) != 0u;
#elif QSB_TAIL_STAGGER == 5
const bool qsb_ffgg = (half != 0) & (batch_size < 0);
#else
const bool qsb_ffgg = (half != 0) | (batch_size >= 0);
#endif
int qsb_encA=0,qsb_encB=0;
auto qsb_tail_fin=[&](const uint64_t *n,const uint64_t *inv) -> QsbPairFin3 {
return qsb_pair_finish3_value(n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],inv[0],inv[1],inv[2],inv[3] QSB_R_PASS_TAIL(u2rx,u2ry));
};
auto qsb_tail_gate=[&](const QsbPairFin3 &f) -> int {
return qsb_pair_gate3_value(f.x[0],f.x[1],f.x[2],f.x[3],f.x[4],f.x[5],f.x[6],f.x[7],f.par);
};
#if QSB_TAIL_PARK == 3

uint64_t invA[5],invB[5],nA[12],nB2[12];
QSB_TREE_MUL(invA,leaf,prodB);
QSB_TREE_MUL(invB,leaf,prodA);
#pragma unroll
for(int k=0;k<12;k++)nA[k]=parkA[k][tid];
#pragma unroll
for(int k=0;k<12;k++)parkA[k][tid]=nB[k];
auto qsb_tail_fin_a=[&]() -> QsbPairFin3 {return qsb_tail_fin(nA,invA);};
auto qsb_tail_fin_b=[&]() -> QsbPairFin3 {
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<12;k++)nB2[k]=parkA[k][tid];
return qsb_tail_fin(nB2,invB);
};
#else
auto qsb_tail_fin_a=[&]() -> QsbPairFin3 {
uint64_t inv[5],n[12];
QSB_TREE_MUL(inv,leaf,prodB);
#pragma unroll
for(int k=0;k<12;k++)n[k]=parkA[k][tid];
return qsb_tail_fin(n,inv);
};
auto qsb_tail_fin_b=[&]() -> QsbPairFin3 {
uint64_t inv[5];
QSB_TREE_MUL(inv,leaf,prodA);
return qsb_tail_fin(nB,inv);
};
#endif
if(qsb_ffgg){
#if QSB_TAIL_PARK == 4

QsbPairFin3 finA;
if(okA){
finA=qsb_tail_fin_a();
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=finA.x[k];
}
if(okB)qsb_encB=qsb_tail_gate(qsb_tail_fin_b());
if(okA){
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<8;k++)finA.x[k]=parkA[k][tid];
qsb_encA=qsb_tail_gate(finA);
}
#elif QSB_TAIL_PARK >= 2


QsbPairFin3 finA,finB;
if(okA)finA=qsb_tail_fin_a();
#if QSB_TAIL_PARK == 3
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<12;k++)nB2[k]=parkA[k][tid];
#endif
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=finA.x[k];
#if QSB_TAIL_PARK == 3
if(okB)finB=qsb_tail_fin(nB2,invB);
#else
if(okB)finB=qsb_tail_fin_b();
#endif
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<8;k++)finA.x[k]=parkA[k][tid];
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=finB.x[k];
if(okA)qsb_encA=qsb_tail_gate(finA);
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<8;k++)finB.x[k]=parkA[k][tid];
if(okB)qsb_encB=qsb_tail_gate(finB);
#else

QsbPairFin3 finA,finB;
if(okA){
finA=qsb_tail_fin_a();
#if QSB_TAIL_PARK == 1
#pragma unroll
for(int k=0;k<8;k++)parkA[k][tid]=finA.x[k];
#endif
}
if(okB)finB=qsb_tail_fin_b();
if(okA){
#if QSB_TAIL_PARK == 1
asm volatile("" ::: "memory");
#pragma unroll
for(int k=0;k<8;k++)finA.x[k]=parkA[k][tid];
#endif
qsb_encA=qsb_tail_gate(finA);
}
if(okB)qsb_encB=qsb_tail_gate(finB);
#endif
}else{
#if QSB_TAIL_WEAVE




{
uint64_t invA[5],n[12];
QSB_TREE_MUL(invA,leaf,prodB);
#pragma unroll
for(int k=0;k<12;k++)n[k]=parkA[k][tid];
#pragma unroll
for(int k=0;k<12;k++)parkA[k][tid]=nB[k];
const QsbPairFin3 finA=qsb_tail_fin(n,invA);
uint64_t invB[5];
QSB_TREE_MUL(invB,leaf,prodA);
const uint32_t qsb_nb_smem=(uint32_t)__cvta_generic_to_shared(&parkA[0][tid]);
const QsbPairWeave3 w=qsb_pair_weave3_value(finA.x[0],finA.x[1],finA.x[2],finA.x[3],finA.x[4],finA.x[5],finA.x[6],finA.x[7],finA.par,
invB[0],invB[1],invB[2],invB[3],qsb_nb_smem QSB_R_PASS_TAIL(u2rx,u2ry));
qsb_encA=w.enc;
QsbPairFin3 finB;
#pragma unroll
for(int k=0;k<8;k++)finB.x[k]=w.x[k];
finB.par=w.par;
qsb_encB=qsb_tail_gate(finB);
}
#else
if(okA)qsb_encA=qsb_tail_gate(qsb_tail_fin_a());
if(okB)qsb_encB=qsb_tail_gate(qsb_tail_fin_b());
#endif
}
if(okA){
int encoded=qsb_encA;
#ifdef QSB_FORCE_EXACT_HIT_CHECK
encoded=1;
#endif
int recid=encoded-1;
if(encoded){
uint32_t pslot=atomicAdd(d_hit_cnt,1);
if(pslot<1024){
d_hit_idx[pslot*4]=(QSB_SC_EA0*(unsigned)QSB_SE_WINDOWS+(unsigned)lane)|((uint32_t)recid<<30);
#if !QSB_HIT_NO_COMBO
for(int i=0;i<6;i++)d_hit_combos[pslot*ZLAB_HIT_REC+i]=QSB_SC_E0->early[i];
for(int i=0;i<3;i++)d_hit_combos[pslot*ZLAB_HIT_REC+6+i]=WIN3[lane][i];
#endif
}
}
}
if(okB){
int encoded=qsb_encB;
#ifdef QSB_FORCE_EXACT_HIT_CHECK
encoded=1;
#endif
int recid=encoded-1;
if(encoded){
uint32_t pslot=atomicAdd(d_hit_cnt,1);
if(pslot<1024){
d_hit_idx[pslot*4]=((QSB_SC_EA0+1u)*(unsigned)QSB_SE_WINDOWS+(unsigned)lane)|((uint32_t)recid<<30);
#if !QSB_HIT_NO_COMBO
for(int i=0;i<6;i++)d_hit_combos[pslot*ZLAB_HIT_REC+i]=QSB_SC_E1->early[i];
for(int i=0;i<3;i++)d_hit_combos[pslot*ZLAB_HIT_REC+6+i]=WIN3[lane][i];
#endif
}
}
}
}
#else
if(okA){
#if ZLAB_K2S3M
uint64_t inv[5],n[12];
#if QSB_DEN_CROSS_PRE == 1 || QSB_DEN_CROSS_PRE == 3
#pragma unroll
for(int j=0;j<5;j++)inv[j]=leaf[j];
#else
QSB_TREE_MUL(inv,leaf,prodB);
#endif
#if QSB_PARK128
#pragma unroll
for(int j=0;j<6;j++){ const ulonglong2 v=parkA2[j][tid]; n[2*j]=v.x; n[2*j+1]=v.y; }
#else
#pragma unroll
for(int k=0;k<12;k++)n[k]=parkA[k][tid];
#endif
int encoded=qsb_pair_tail3_value(n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],inv[0],inv[1],inv[2],inv[3] QSB_R_PASS_TAIL(u2rx,u2ry));
#else
uint64_t inv[5],m1[4],m2[4];
QSB_TREE_MUL(inv,leaf,prodB);
#pragma unroll
for(int k=0;k<4;k++){m1[k]=parkA[k][tid];m2[k]=parkA[4+k][tid];}
int encoded=qsb_pair_tail_value(m1[0],m1[1],m1[2],m1[3],m2[0],m2[1],m2[2],m2[3],inv[0],inv[1],inv[2],inv[3],u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
#endif
#ifdef QSB_FORCE_EXACT_HIT_CHECK
encoded=1;
#endif
int recid=encoded-1;
if(encoded){
uint32_t pslot=atomicAdd(d_hit_cnt,1);
if(pslot<1024){
d_hit_idx[pslot*4]=(QSB_SC_EA0*(unsigned)QSB_SE_WINDOWS+(unsigned)lane)|((uint32_t)recid<<30);
#if !QSB_HIT_NO_COMBO
for(int i=0;i<6;i++)d_hit_combos[pslot*ZLAB_HIT_REC+i]=QSB_SC_E0->early[i];
for(int i=0;i<3;i++)d_hit_combos[pslot*ZLAB_HIT_REC+6+i]=WIN3[lane][i];
#endif
}
}
}
if(okB){
uint64_t inv[5];
#if QSB_DEN_CROSS_PRE == 1 || QSB_DEN_CROSS_PRE == 2
#pragma unroll
for(int j=0;j<5;j++)inv[j]=leaf[j];
#else
QSB_TREE_MUL(inv,leaf,prodA);
#endif
#if ZLAB_K2S3M
int encoded=qsb_pair_tail3_value(nB[0],nB[1],nB[2],nB[3],nB[4],nB[5],nB[6],nB[7],nB[8],nB[9],nB[10],nB[11],inv[0],inv[1],inv[2],inv[3] QSB_R_PASS_TAIL(u2rx,u2ry));
#else
int encoded=qsb_pair_tail_value(m1B[0],m1B[1],m1B[2],m1B[3],m2B[0],m2B[1],m2B[2],m2B[3],inv[0],inv[1],inv[2],inv[3],u2rx[0],u2rx[1],u2rx[2],u2rx[3],u2ry[0],u2ry[1],u2ry[2],u2ry[3]);
#endif
#ifdef QSB_FORCE_EXACT_HIT_CHECK
encoded=1;
#endif
int recid=encoded-1;
if(encoded){
uint32_t pslot=atomicAdd(d_hit_cnt,1);
if(pslot<1024){
d_hit_idx[pslot*4]=((QSB_SC_EA0+1u)*(unsigned)QSB_SE_WINDOWS+(unsigned)lane)|((uint32_t)recid<<30);
#if !QSB_HIT_NO_COMBO
for(int i=0;i<6;i++)d_hit_combos[pslot*ZLAB_HIT_REC+i]=QSB_SC_E1->early[i];
for(int i=0;i<3;i++)d_hit_combos[pslot*ZLAB_HIT_REC+6+i]=WIN3[lane][i];
#endif
}
}
}
#endif
QSB_DP_STAMP(6);
#else

int idx = blockIdx.x * blockDim.x + threadIdx.x;

const int easy_flag=0,single_hash_flag=1,calibrate_flag=0;

if(blockIdx.x*blockDim.x>=batch_size)return;
int active=idx<batch_size;











#if ZLAB_TRIM
const epoch_desc_t *se_desc = d_epochs + blockIdx.x;
uint32_t state[8];
for (int i = 0; i < 8; i++) state[i] = se_desc->mid[i];
qsb_scheduled_window_hash(state, se_desc, threadIdx.x, d_first+(size_t)blockIdx.x*QSB_FIRST_SLOTS*8);
#else
uint8_t skip[MAX_T];
const epoch_desc_t *se_desc = NULL;
if (fast_inc == QSB_SE_N_INC) {





se_desc = d_epochs + blockIdx.x;

} else if (d_combos == NULL) {
unrank_combo(enum_base + (uint64_t)(active?idx:0), n_pool - window_start, t_win, skip + s_early);
for (int i = 0; i < t_win; ++i) skip[s_early + i] += window_start;
for (int i = 0; i < s_early; ++i) skip[i] = d_early[i];
} else {
for (int i = 0; i < t_sel; i++)
skip[i] = d_combos[(active ? idx : 0) * t_sel + i];
}






uint32_t state[8];
if (se_desc) {
for (int i = 0; i < 8; i++) state[i] = se_desc->mid[i];
} else {
for (int i = 0; i < 8; i++) state[i] = d_midstate[i];
}

if (fast_inc == QSB_SE_N_INC) {
qsb_scheduled_window_hash(state, se_desc, threadIdx.x, d_first+(size_t)blockIdx.x*QSB_FIRST_SLOTS*8);
} else if (fast_inc == QSB_FAST_N_INC) {


bool cached=d_combos==NULL && qsb_prefix_eligible(n_pool,window_start,t_win,
fast_inc,prefix_remainder_len);
qsb_fast_window_hash(state,skip,t_win,s_early,window_start,cached,d_const_words);
} else {
uint32_t curW[16];
uint8_t *cur = (uint8_t *)curW;
uint32_t blk[16];
int cur_pos = 0;

for (int i = 0; i < prefix_remainder_len; i++) {
cur[cur_pos++] = d_prefix_remainder[i];
if (cur_pos == 64) {
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}
{


int sel = s_early;
for (int i = window_start; i < n_pool; i++) {
if (sel < t_sel && skip[sel] == i) { sel++; continue; }
const uint8_t *row = d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE;
for (int b = 0; b < SIG_PUSH_SIZE; b++) {
cur[cur_pos++] = row[b];
if (cur_pos == 64) {
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}
}
}
for (int i = 0; i < tail_len; i++) {
cur[cur_pos++] = d_tail[i];
if (cur_pos == 64) {
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}
for (int i = 0; i < tx_suffix_len; i++) {
cur[cur_pos++] = d_tx_suffix[i];
if (cur_pos == 64) {
for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
_SHA256Transform(state, blk);
cur_pos = 0;
}
}


uint32_t lastW[32];
uint8_t *last_block = (uint8_t *)lastW;
int rem = cur_pos;
memset(last_block, 0, 128);
memcpy(last_block, cur, rem);
last_block[rem] = 0x80;
int nblk = (rem < 56) ? 1 : 2;
uint64_t bit_len = (uint64_t)total_preimage_len * 8;
int last = nblk * 64 - 8;
last_block[last]=(bit_len>>56)&0xFF; last_block[last+1]=(bit_len>>48)&0xFF;
last_block[last+2]=(bit_len>>40)&0xFF; last_block[last+3]=(bit_len>>32)&0xFF;
last_block[last+4]=(bit_len>>24)&0xFF; last_block[last+5]=(bit_len>>16)&0xFF;
last_block[last+6]=(bit_len>>8)&0xFF; last_block[last+7]=bit_len&0xFF;

for (int b = 0; b < nblk; b++) {
uint32_t blk2[16];
for (int i = 0; i < 16; i++) blk2[i] = bswap32(lastW[b*16+i]);
_SHA256Transform(state, blk2);
}
}
#endif




uint32_t b2[16];
for (int i=0;i<8;i++) b2[i]=state[i];
b2[8]=0x80000000;
for (int i=9;i<15;i++) b2[i]=0;
b2[15]=0x00000100;
uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
_SHA256Transform(s2, b2);





uint64_t z[4];
z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];





uint64_t qx[4],qy[4],qzz[4],qzzz[4];
_FixedBaseSignedXYZZStream(qx,qy,qzz,qzzz,z,d_gt);

uint64_t u2rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
uint64_t u2ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};


uint64_t prod[5];
qsb_xyzz_finish_prepare(qx,qzz,qzzz,u2rx,prod);
bool usable = active && ((prod[0]|prod[1]|prod[2]|prod[3]) != 0);



if(!usable){prod[0]=1;prod[1]=prod[2]=prod[3]=prod[4]=0;}
qsb_block_inverse_tree(prod);
if(!usable)return;
uint64_t q1x[4],q2x[4];
uint32_t y_parities = qsb_xyzz_finish_precomputed(qy,qzz,qzzz,prod,u2rx,u2ry,q1x,q2x);

int v=0, hash_choice=0, recid=0;
#if ZLAB_PAIRSHA
{
uint32_t pw[2][16];
for(int ri=0;ri<2;ri++){
uint64_t sx0=ri ? q2x[0] : q1x[0];
uint64_t sx1=ri ? q2x[1] : q1x[1];
uint64_t sx2=ri ? q2x[2] : q1x[2];
uint64_t sx3=ri ? q2x[3] : q1x[3];
uint32_t x32[8]={(uint32_t)sx0,(uint32_t)(sx0>>32),(uint32_t)sx1,(uint32_t)(sx1>>32),
(uint32_t)sx2,(uint32_t)(sx2>>32),(uint32_t)sx3,(uint32_t)(sx3>>32)};
uint8_t prefix_byte = 0x2+(uint8_t)((y_parities>>ri)&1u);
uint32_t *pb=pw[ri];
pb[0]=__byte_perm(x32[7],prefix_byte,0x4321);
pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
}
uint32_t hs0[8],hs1[8];
zlab_sha256_pair_h0(pw[0],pw[1],hs0,hs1);
if(gpu_bench_valid_words(hs0)){v=1;recid=0;}
else if(gpu_bench_valid_words(hs1)){v=1;recid=1;}
}
#else
for(int ri=0;ri<2&&!v;ri++){
uint64_t sx0=ri ? q2x[0] : q1x[0];
uint64_t sx1=ri ? q2x[1] : q1x[1];
uint64_t sx2=ri ? q2x[2] : q1x[2];
uint64_t sx3=ri ? q2x[3] : q1x[3];
uint32_t x32[8]={(uint32_t)sx0,(uint32_t)(sx0>>32),(uint32_t)sx1,(uint32_t)(sx1>>32),
(uint32_t)sx2,(uint32_t)(sx2>>32),(uint32_t)sx3,(uint32_t)(sx3>>32)};
uint32_t pb[16];
uint8_t prefix_byte = 0x2+(uint8_t)((y_parities>>ri)&1u);
pb[0]=__byte_perm(x32[7],prefix_byte,0x4321);
pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);


int vv;
if (calibrate_flag || easy_flag) {
uint8_t h[32];
for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
vv = calibrate_flag ? gpu_is_der_relaxed(h,32) : gpu_is_der_easy(h,32);
} else {
vv = gpu_bench_valid_words(hs);
}
if(vv){ v=1;hash_choice=0;recid=ri; break; }



if (single_hash_flag) continue;
uint8_t h[32];
for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
uint8_t pp[64];memset(pp,0,64);memcpy(pp,h,32);pp[32]=0x80;pp[62]=1;pp[63]=0;
uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
if (calibrate_flag || easy_flag) {
uint8_t h2[32];
for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
vv = calibrate_flag ? gpu_is_der_relaxed(h2,32) : gpu_is_der_easy(h2,32);
} else {
vv = gpu_bench_valid_words(h2s);
}
if(vv){ v=1;hash_choice=1;recid=ri; break; }
}
#endif





if(v){uint32_t p=atomicAdd(d_hit_cnt,1);
if(p<1024) {
#if ZLAB_HITPATH
if(se_desc) {

d_hit_idx[p*4]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);
for(int i=0;i<6;i++)d_hit_combos[p*ZLAB_HIT_REC+i]=se_desc->early[i];
for(int i=0;i<3;i++)d_hit_combos[p*ZLAB_HIT_REC+6+i]=WIN3[threadIdx.x][i];
} else
#endif
#if ZLAB_TRIM
#if ZLAB_HITPATH
{}
#else
{
d_hit_idx[p]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);
for(int i=0;i<6;i++)d_hit_combos[p*MAX_T+i]=se_desc->early[i];
for(int i=0;i<3;i++)d_hit_combos[p*MAX_T+6+i]=WIN3[threadIdx.x][i];
}
#endif
#else
{
d_hit_idx[p]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);
if(se_desc) {
for(int i=0;i<6;i++)d_hit_combos[p*MAX_T+i]=se_desc->early[i];
for(int i=0;i<3;i++)d_hit_combos[p*MAX_T+6+i]=WIN3[threadIdx.x][i];
} else {
for(int i=0;i<t_sel;i++)d_hit_combos[p*MAX_T+i]=skip[i];
}
}
#endif
}
}

#endif
}












































#ifndef QSB_GT_BATCH
#define QSB_GT_BATCH 1
#endif
#ifndef QSB_GT_PREFIX_CACHE
#define QSB_GT_PREFIX_CACHE 0
#endif
#ifndef QSB_GT_PHASE_AUDIT
#define QSB_GT_PHASE_AUDIT 0
#endif
#if QSB_GT_PHASE_AUDIT && !(QSB_S3 && QSB_GT_HEAL && QSB_STARTUP_TRIM)
#error "GT phase audit requires the production S3 heal and trimmed spot-check paths"
#endif
#if QSB_GT_PREFIX_CACHE != 0 && QSB_GT_PREFIX_CACHE != 1
#error "QSB_GT_PREFIX_CACHE must be 0 or 1"
#endif
#if QSB_GT_BATCH != 0 && QSB_GT_BATCH != 1
#error "QSB_GT_BATCH must be 0 or 1"
#endif
#if QSB_S3







#if QSB_GT_BATCH

#define GT_BATCH_N 16

#define GT_BATCH_THREADS \
((((unsigned long long)GT_TOTAL_ENTRIES + 32ull * GT_BATCH_N - 1) / (32ull * GT_BATCH_N)) * 32ull)




__device__ __forceinline__ void gt_batch_store(uint8_t *gTable, size_t off,
const uint64_t x[4], const uint64_t y[4]) {
ulonglong2 *r = (ulonglong2 *)(gTable + off);
__stcs(r, make_ulonglong2(x[0], x[1]));
__stcs(r + 1, make_ulonglong2(x[2], x[3]));
__stcs(r + 2, make_ulonglong2(y[0], y[1]));
__stcs(r + 3, make_ulonglong2(y[2], y[3]));
}
__global__ void __launch_bounds__(256, 2) kernel_build_gtable(
const uint64_t * __restrict__ d_L,
const uint64_t * __restrict__ d_H,
const uint64_t * __restrict__ d_H2,
uint8_t * __restrict__ gTable)
{
const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
const uint64_t t0 = (g >> 5) * (32u * GT_BATCH_N) + (g & 31);
if (t0 >= GT_TOTAL_ENTRIES) return;
uint64_t X[GT_BATCH_N][4], Y[GT_BATCH_N][4], Z[GT_BATCH_N][4], A[GT_BATCH_N][4];
uint32_t rec[GT_BATCH_N];
int n = 0;
#if QSB_GT_PREFIX_CACHE
int cached_ch=-1,cached_hi=-1,cached_h2=-1;
uint64_t cached_x[4],cached_y[4],cached_z[4];
#endif



#pragma unroll 1
for (int j = 0; j < GT_BATCH_N; j++) {
const uint64_t t = t0 + 32u * (uint64_t)j;
if (t >= GT_TOTAL_ENTRIES) break;
int ch=-1;
#pragma unroll
for(int c=0;c<GT_CHUNKS;c++)
if(t>=gt_offset(c) && t<(uint64_t)gt_offset(c)+gt_entries(c)) ch=c;
if(ch<0) continue;
int d=(int)(t-gt_offset(ch));
int m = ch==0?d:2*d+1;
int h2 = m >> 24, hi = (m >> 12) & 4095, lo = m & 4095;

const uint64_t *Hp = d_H + ((size_t)ch * GT_HI + hi) * 8;
const uint64_t *Lp = d_L + ((size_t)ch * GT_LO + lo) * 8;
const size_t off = ((size_t)gt_offset(ch) + d) * 64;

if (hi == 0 && h2 == 0) {
uint64_t rx[4], ry[4];
for (int k = 0; k < 4; k++) { rx[k] = Lp[k]; ry[k] = Lp[4 + k]; }
gt_batch_store(gTable, off, rx, ry);
continue;
}
uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0}, qx[4], qy[4];
const uint64_t *Sp = h2 ? d_H2 + ((size_t)ch * GT_H2 + h2) * 8 : Hp;
for (int k = 0; k < 4; k++) { px[k] = Sp[k]; py[k] = Sp[4 + k]; }
if (h2 != 0 && hi != 0) {
#if QSB_GT_PREFIX_CACHE
if(cached_ch!=ch || cached_hi!=hi || cached_h2!=h2){
#endif
for (int k = 0; k < 4; k++) { qx[k] = Hp[k]; qy[k] = Hp[4 + k]; }
_PointAddSecp256k1(px, py, pz, qx, qy);
#if QSB_GT_PREFIX_CACHE
for(int k=0;k<4;k++){cached_x[k]=px[k];cached_y[k]=py[k];cached_z[k]=pz[k];}
cached_ch=ch;cached_hi=hi;cached_h2=h2;
}else{
for(int k=0;k<4;k++){px[k]=cached_x[k];py[k]=cached_y[k];pz[k]=cached_z[k];}
}
#endif
}
for (int k = 0; k < 4; k++) { qx[k] = Lp[k]; qy[k] = Lp[4 + k]; }
_PointAddSecp256k1(px, py, pz, qx, qy);
for (int k = 0; k < 4; k++) { X[n][k] = px[k]; Y[n][k] = py[k]; Z[n][k] = pz[k]; }
if (n == 0) { for (int k = 0; k < 4; k++) A[0][k] = pz[k]; }
else _ModMult(A[n], A[n - 1], pz);
rec[n] = (uint32_t)t;
n++;
}
if (n == 0) return;
uint64_t inv[5] = {A[n - 1][0], A[n - 1][1], A[n - 1][2], A[n - 1][3], 0};
_ModInv(inv);


#pragma unroll 1
for (int j = n - 1; j >= 0; j--) {
uint64_t zi[5];
if (j > 0) {
_ModMult(zi, inv, A[j - 1]);
_ModMult(inv, Z[j]);
} else {
for (int k = 0; k < 4; k++) zi[k] = inv[k];
}

uint64_t e[4];
_ModMult(e, Z[j], zi);
const bool canon = !((zi[1] & zi[2] & zi[3]) == 0xFFFFFFFFFFFFFFFFULL &&
zi[0] >= 0xFFFFFFFEFFFFFC2FULL);
if (!canon || e[0] != 1 || (e[1] | e[2] | e[3]) != 0) {

for (int k = 0; k < 4; k++) zi[k] = Z[j][k];
zi[4] = 0;
_ModInv(zi);
}
_ModMult(X[j], zi); _ModMult(Y[j], zi);
gt_batch_store(gTable, (size_t)rec[j] * 64, X[j], Y[j]);
}
}
#else
__global__ void kernel_build_gtable(
const uint64_t * __restrict__ d_L,
const uint64_t * __restrict__ d_H,
const uint64_t * __restrict__ d_H2,
uint8_t * __restrict__ gTable)
{
uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (t >= GT_TOTAL_ENTRIES) return;
int ch=-1;
#pragma unroll
for(int c=0;c<GT_CHUNKS;c++)
if(t>=gt_offset(c) && t<(uint64_t)gt_offset(c)+gt_entries(c)) ch=c;
if(ch<0) return;
int d=(int)(t-gt_offset(ch));
int m = ch==0?d:2*d+1;
int h2 = m >> 24, hi = (m >> 12) & 4095, lo = m & 4095;

const uint64_t *Hp = d_H + ((size_t)ch * GT_HI + hi) * 8;
const uint64_t *Lp = d_L + ((size_t)ch * GT_LO + lo) * 8;

uint64_t rx[4], ry[4];
if (hi == 0 && h2 == 0) {
for (int k = 0; k < 4; k++) { rx[k] = Lp[k]; ry[k] = Lp[4 + k]; }
} else {
uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0}, qx[4], qy[4];


const uint64_t *Sp = h2 ? d_H2 + ((size_t)ch * GT_H2 + h2) * 8 : Hp;
for (int k = 0; k < 4; k++) { px[k] = Sp[k]; py[k] = Sp[4 + k]; }
if (h2 != 0 && hi != 0) {
for (int k = 0; k < 4; k++) { qx[k] = Hp[k]; qy[k] = Hp[4 + k]; }
_PointAddSecp256k1(px, py, pz, qx, qy);
}
for (int k = 0; k < 4; k++) { qx[k] = Lp[k]; qy[k] = Lp[4 + k]; }
_PointAddSecp256k1(px, py, pz, qx, qy);
_ModInv(pz);
_ModMult(px, pz); _ModMult(py, pz);
for (int k = 0; k < 4; k++) { rx[k] = px[k]; ry[k] = py[k]; }
}
size_t off = ((size_t)gt_offset(ch) + d) * 64;
memcpy(gTable + off, rx, 32);
memcpy(gTable + off + 32, ry, 32);
}
#endif
#if QSB_YOFF_S


__global__ void kernel_gt_offset_y(uint8_t * __restrict__ gTable) {
const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (t >= GT_TOTAL_ENTRIES) return;
ulonglong2 *r = (ulonglong2 *)(gTable + t * 64 + 32);
ulonglong2 a = r[0], b = r[1];
asm("add.cc.u64 %0, %0, 0x800001E8;\n\taddc.cc.u64 %1, %1, 0;\n\taddc.cc.u64 %2, %2, 0;\n\taddc.u64 %3, %3, 0;"
: "+l"(a.x), "+l"(a.y), "+l"(b.x), "+l"(b.y));
r[0] = a; r[1] = b;
}
#endif
#if QSB_GT_HEAL




__device__ __forceinline__ void gt_heal_canon(uint64_t a[4]) {

unsigned __int128 c = (unsigned __int128)a[0] + 0x1000003D1ULL;
const uint64_t t0 = (uint64_t)c; c = (c >> 64) + a[1];
const uint64_t t1 = (uint64_t)c; c = (c >> 64) + a[2];
const uint64_t t2 = (uint64_t)c; c = (c >> 64) + a[3];
const uint64_t t3 = (uint64_t)c;
if ((uint64_t)(c >> 64)) { a[0] = t0; a[1] = t1; a[2] = t2; a[3] = t3; }
}
__global__ void kernel_gt_heal_scan(const uint8_t * __restrict__ gTable,
const uint64_t * __restrict__ bprime,
unsigned *flags, unsigned cap) {
const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (t >= GT_TOTAL_ENTRIES) return;
const uint64_t *r = (const uint64_t *)(gTable + t * 64);
uint64_t x[4] = {r[0], r[1], r[2], r[3]}, y[4] = {r[4], r[5], r[6], r[7]};
uint64_t b[4] = {bprime[0], bprime[1], bprime[2], bprime[3]};
uint64_t lhs[4], x2[4], rhs[4];
_ModSqr(lhs, y); _ModSqr(x2, x); _ModMult(rhs, x2, x); _ModAdd256(rhs, rhs, b);
gt_heal_canon(lhs); gt_heal_canon(rhs);
if (lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[2] != rhs[2] || lhs[3] != rhs[3]) {
const unsigned slot = atomicAdd(&flags[0], 1u);
if (slot < cap) flags[1 + slot] = (unsigned)t;
}
}
#endif
#else
__global__ void kernel_build_gtable(
const uint64_t * __restrict__ d_L,
const uint64_t * __restrict__ d_H,
uint8_t * __restrict__ gTable)
{
uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (t >= GT_TOTAL_ENTRIES) return;
#if ZLAB_T14
int ch=t<((uint64_t)GT_BIG<<18)?(int)(t>>18):GT_BIG+(int)((t-((uint64_t)GT_BIG<<18))>>17);
#else
int ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);
#endif
int d=(int)(t-gt_offset(ch));
int m = 2*d + 1;
int hi = m >> 8, lo = m & 255;

const uint64_t *Hp = d_H + ((size_t)ch * GT_HI + hi) * 8;
const uint64_t *Lp = d_L + ((size_t)ch * GT_LO + lo) * 8;

uint64_t rx[4], ry[4];
if (hi == 0) {
for (int k = 0; k < 4; k++) { rx[k] = Lp[k]; ry[k] = Lp[4 + k]; }
} else {
uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0}, qx[4], qy[4];
for (int k = 0; k < 4; k++) {
px[k] = Hp[k]; py[k] = Hp[4 + k];
qx[k] = Lp[k]; qy[k] = Lp[4 + k];
}
_PointAddSecp256k1(px, py, pz, qx, qy);
_ModInv(pz);
_ModMult(px, pz); _ModMult(py, pz);
for (int k = 0; k < 4; k++) { rx[k] = px[k]; ry[k] = py[k]; }
}


size_t off = ((size_t)gt_offset(ch) + d) * 64;
memcpy(gTable + off, rx, 32);
memcpy(gTable + off + 32, ry, 32);
}
#endif





extern "C" {
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
}


static void gt_point_to_limbs(EC_GROUP *grp, EC_POINT *pt, BIGNUM *x, BIGNUM *y,
const BIGNUM *alpha, const BIGNUM *beta,
const BIGNUM *field_p, BN_CTX *ctx, uint64_t out[8]) {
uint8_t xb[32], yb[32];
memset(xb, 0, 32); memset(yb, 0, 32);
EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
BN_mod_mul(x,x,alpha,field_p,ctx);
BN_mod_mul(y,y,beta, field_p,ctx);
BN_bn2bin(x, xb + (32 - BN_num_bytes(x)));
BN_bn2bin(y, yb + (32 - BN_num_bytes(y)));
for (int j = 0; j < 16; j++) { uint8_t t = xb[j]; xb[j] = xb[31-j]; xb[31-j] = t; }
for (int j = 0; j < 16; j++) { uint8_t t = yb[j]; yb[j] = yb[31-j]; yb[31-j] = t; }
memcpy(out, xb, 32);
memcpy(out + 4, yb, 32);
}

#if QSB_S3




static void gt_batch_ladder(EC_GROUP *grp, const EC_POINT *step, int count,
uint64_t *out, BIGNUM *x, BIGNUM *y,
const BIGNUM *alpha, const BIGNUM *beta,
const BIGNUM *field_p, BN_CTX *ctx) {
EC_POINT *points[GT_HI];
if(count<1 || count>=GT_HI) { fprintf(stderr,"Invalid ladder size\n");exit(2); }
for(int i=0;i<count;i++) {
points[i]=EC_POINT_new(grp);
if(!points[i]) { fprintf(stderr,"Ladder allocation failed\n");exit(2); }
int ok=i==0 ? EC_POINT_copy(points[i],step)
: EC_POINT_add(grp,points[i],points[i-1],step,ctx);
if(!ok) { fprintf(stderr,"Ladder addition failed\n");exit(2); }
}
if(!EC_POINTs_make_affine(grp,(size_t)count,points,ctx)) {
fprintf(stderr,"Ladder batch normalization failed\n");exit(2);
}
for(int i=0;i<count;i++) {
gt_point_to_limbs(grp,points[i],x,y,alpha,beta,field_p,ctx,
out+(size_t)(i+1)*8);
EC_POINT_free(points[i]);
}
}



static void gt_biased_ladder(EC_GROUP *grp, const EC_POINT *first,
const EC_POINT *step, int count,
uint64_t *out, BIGNUM *x, BIGNUM *y,
const BIGNUM *alpha, const BIGNUM *beta,
const BIGNUM *field_p, BN_CTX *ctx) {
EC_POINT *points[GT_HI];
if(count<1 || count>GT_HI) { fprintf(stderr,"Invalid biased ladder size\n");exit(2); }
for(int i=0;i<count;i++) {
points[i]=EC_POINT_new(grp);
if(!points[i]) { fprintf(stderr,"Biased ladder allocation failed\n");exit(2); }
int ok=i==0 ? EC_POINT_copy(points[i],first)
: EC_POINT_add(grp,points[i],points[i-1],step,ctx);
if(!ok) { fprintf(stderr,"Biased ladder addition failed\n");exit(2); }
}
if(!EC_POINTs_make_affine(grp,(size_t)count,points,ctx)) {
fprintf(stderr,"Biased ladder normalization failed\n");exit(2);
}
for(int i=0;i<count;i++) {
gt_point_to_limbs(grp,points[i],x,y,alpha,beta,field_p,ctx,
out+(size_t)i*8);
EC_POINT_free(points[i]);
}
}




static void gt_build_ladders(uint64_t *hL, uint64_t *hH, uint64_t *hH2,
const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x=BN_new(),*y=BN_new(),*factor=BN_new(),*order=BN_new(),
*nri=BN_new(),*bscal=BN_new(),*field_p=BN_new(),
*alpha=BN_new(),*beta=BN_new(),*bias=BN_new();
EC_POINT *base=EC_POINT_new(grp),*step=EC_POINT_new(grp),*first=EC_POINT_new(grp);
EC_GROUP_get_order(grp,order,ctx);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
BN_lebin2bn(neg_r_inv,32,nri);
BN_set_word(bias,170559770); BN_lshift(bias,bias,99); BN_sub_word(bias,1u<<17);
memset(hH2,0,(size_t)GT_CHUNKS*GT_H2*8*sizeof(uint64_t));
memset(hL,0,(size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t));
memset(hH,0,(size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t));
for(int ch=0;ch<GT_CHUNKS;ch++) {
if(ch==0) {

EC_POINT_mul(grp,base,nri,NULL,NULL,ctx);
BN_mod_mul(bscal,bias,nri,order,ctx);
EC_POINT_mul(grp,first,bscal,NULL,NULL,ctx);
gt_biased_ladder(grp,first,base,GT_LO,
hL,x,y,alpha,beta,field_p,ctx);
} else {

BN_one(factor); BN_lshift(factor,factor,gt_shift(ch)-1);
BN_mod_mul(bscal,factor,nri,order,ctx);
EC_POINT_mul(grp,base,bscal,NULL,NULL,ctx);
gt_batch_ladder(grp,base,GT_LO-1,
hL+(size_t)ch*GT_LO*8,x,y,alpha,beta,field_p,ctx);
}
BN_set_word(factor,GT_LO);
EC_POINT_mul(grp,step,NULL,base,factor,ctx);
unsigned max_m=ch==0?gt_entries(ch)-1:2*(gt_entries(ch)-1)+1;

int high=(int)(max_m>>12);
if(high>GT_HI-1) high=GT_HI-1;
gt_batch_ladder(grp,step,high,
hH+(size_t)ch*GT_HI*8,x,y,alpha,beta,field_p,ctx);
int high2=(int)(max_m>>24);
if(high2>=GT_H2) { fprintf(stderr,"Invalid H2 ladder size\n");exit(2); }
if(high2>0) {
BN_set_word(factor,1u<<24);
EC_POINT_mul(grp,step,NULL,base,factor,ctx);
gt_batch_ladder(grp,step,high2,
hH2+(size_t)ch*GT_H2*8,x,y,alpha,beta,field_p,ctx);
}
}
BN_free(x);BN_free(y);BN_free(factor);BN_free(order);BN_free(nri);
BN_free(bscal);BN_free(field_p);BN_free(alpha);BN_free(beta);BN_free(bias);
EC_POINT_free(base);EC_POINT_free(step);EC_POINT_free(first);
EC_GROUP_free(grp);BN_CTX_free(ctx);
}

#if QSB_STARTUP_THREADS || QSB_S1_LADDER_SELFTEST
#include <thread>
#include <vector>


static void gt_build_ladders_seg(int ch, uint64_t *hL, uint64_t *hH, uint64_t *hH2,
const uint8_t *neg_r_inv,
const uint64_t *alpha_le, const uint64_t *beta_le) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x=BN_new(),*y=BN_new(),*factor=BN_new(),*order=BN_new(),
*nri=BN_new(),*bscal=BN_new(),*field_p=BN_new(),
*alpha=BN_new(),*beta=BN_new(),*bias=BN_new();
EC_POINT *base=EC_POINT_new(grp),*step=EC_POINT_new(grp),*first=EC_POINT_new(grp);
EC_GROUP_get_order(grp,order,ctx);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
BN_lebin2bn(neg_r_inv,32,nri);
BN_set_word(bias,170559770); BN_lshift(bias,bias,99); BN_sub_word(bias,1u<<17);
if(ch==0) {

EC_POINT_mul(grp,base,nri,NULL,NULL,ctx);
BN_mod_mul(bscal,bias,nri,order,ctx);
EC_POINT_mul(grp,first,bscal,NULL,NULL,ctx);
gt_biased_ladder(grp,first,base,GT_LO,
hL,x,y,alpha,beta,field_p,ctx);
} else {

BN_one(factor); BN_lshift(factor,factor,gt_shift(ch)-1);
BN_mod_mul(bscal,factor,nri,order,ctx);
EC_POINT_mul(grp,base,bscal,NULL,NULL,ctx);
gt_batch_ladder(grp,base,GT_LO-1,
hL+(size_t)ch*GT_LO*8,x,y,alpha,beta,field_p,ctx);
}
BN_set_word(factor,GT_LO);
EC_POINT_mul(grp,step,NULL,base,factor,ctx);
unsigned max_m=ch==0?gt_entries(ch)-1:2*(gt_entries(ch)-1)+1;
int high=(int)(max_m>>12);
if(high>GT_HI-1) high=GT_HI-1;
gt_batch_ladder(grp,step,high,
hH+(size_t)ch*GT_HI*8,x,y,alpha,beta,field_p,ctx);
int high2=(int)(max_m>>24);
if(high2>=GT_H2) { fprintf(stderr,"Invalid H2 ladder size\n");exit(2); }
if(high2>0) {
BN_set_word(factor,1u<<24);
EC_POINT_mul(grp,step,NULL,base,factor,ctx);
gt_batch_ladder(grp,step,high2,
hH2+(size_t)ch*GT_H2*8,x,y,alpha,beta,field_p,ctx);
}
BN_free(x);BN_free(y);BN_free(factor);BN_free(order);BN_free(nri);
BN_free(bscal);BN_free(field_p);BN_free(alpha);BN_free(beta);BN_free(bias);
EC_POINT_free(base);EC_POINT_free(step);EC_POINT_free(first);
EC_GROUP_free(grp);BN_CTX_free(ctx);
}

static void gt_build_ladders_mt(uint64_t *hL, uint64_t *hH, uint64_t *hH2,
const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
memset(hH2,0,(size_t)GT_CHUNKS*GT_H2*8*sizeof(uint64_t));
memset(hL,0,(size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t));
memset(hH,0,(size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t));
std::vector<std::thread> th;
for(int ch=0;ch<GT_CHUNKS;ch++)
th.emplace_back(gt_build_ladders_seg,ch,hL,hH,hH2,neg_r_inv,alpha_le,beta_le);
for(auto &t: th) t.join();
}
#endif


static void gt_table_scalar(BIGNUM *k,int ch,unsigned index) {
if(ch==0) {
BN_set_word(k,170559770); BN_lshift(k,k,99);
BN_sub_word(k,1u<<17); BN_add_word(k,index);
} else {
BN_one(k); BN_lshift(k,k,gt_shift(ch)-1);
BN_mul_word(k,(BN_ULONG)(2*index+1));
}
}







static void gt_spot_sample(int t, unsigned *seed, int *ch_out, int *i_out) {
int ch, i;
if (t < GT_CHUNKS * 8) {
ch = t / 8;
const int e = (int)gt_entries(ch), z = ch==0;
const int corner[8] = {0, 1, 2, z?4095:2047, z?4096:2048,
z?(1<<24):(1<<23), z?(1<<24)+4096:(1<<23)+2048, e - 1};
i = corner[t % 8] < e ? corner[t % 8] : e - 1;
} else {
*seed = *seed * 1664525u + 1013904223u;
ch = (int)(*seed >> 28) % GT_CHUNKS;
*seed = *seed * 1664525u + 1013904223u;
i = (int)(*seed % gt_entries(ch));
}
*ch_out = ch; *i_out = i;
}


static int gt_spot_check(const uint8_t *gTable, int samples,
const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4],
const uint8_t *gathered) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *order = BN_new(),
*nri = BN_new(), *field_p=BN_new(), *alpha=BN_new(), *beta=BN_new();
EC_POINT *pt = EC_POINT_new(grp);
uint64_t want[8];
int ok = 1;
unsigned seed = 0x9e3779b9u;
EC_GROUP_get_order(grp, order, ctx);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
BN_lebin2bn(neg_r_inv, 32, nri);
for (int t = 0; t < samples && ok; t++) {
int ch, i;
gt_spot_sample(t, &seed, &ch, &i);
gt_table_scalar(k, ch, (unsigned)i);
BN_mod_mul(k, k, nri, order, ctx);
EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
gt_point_to_limbs(grp, pt, x, y, alpha,beta,field_p,ctx,want);
const uint8_t *rec = gathered ? gathered + (size_t)t * 64
: gTable + ((size_t)gt_offset(ch) + i) * 64;
if (memcmp(rec, want, 32) != 0 ||
memcmp(rec + 32, want + 4, 32) != 0) {
fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
ok = 0;
}
}
BN_free(x); BN_free(y); BN_free(k); BN_free(order); BN_free(nri);
BN_free(field_p); BN_free(alpha); BN_free(beta);
EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
return ok;
}

#if QSB_STARTUP_THREADS || QSB_S1_LADDER_SELFTEST



static void gt_spot_check_range(int t0, int t1, const int *chs, const int *is, const uint8_t *neg_r_inv,
const uint64_t *alpha_le, const uint64_t *beta_le,
const uint8_t *gathered, int *first_bad) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *order = BN_new(),
*nri = BN_new(), *field_p=BN_new(), *alpha=BN_new(), *beta=BN_new();
EC_POINT *pt = EC_POINT_new(grp);
uint64_t want[8];
EC_GROUP_get_order(grp, order, ctx);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
BN_lebin2bn(neg_r_inv, 32, nri);
*first_bad = -1;
for (int t = t0; t < t1; t++) {
gt_table_scalar(k, chs[t], (unsigned)is[t]);
BN_mod_mul(k, k, nri, order, ctx);
EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
gt_point_to_limbs(grp, pt, x, y, alpha,beta,field_p,ctx,want);
const uint8_t *rec = gathered + (size_t)t * 64;
if (memcmp(rec, want, 32) != 0 || memcmp(rec + 32, want + 4, 32) != 0) { *first_bad = t; break; }
}
BN_free(x); BN_free(y); BN_free(k); BN_free(order); BN_free(nri);
BN_free(field_p); BN_free(alpha); BN_free(beta);
EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
}
static int gt_spot_check_mt(int samples, const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4], const uint8_t *gathered) {
std::vector<int> chs(samples), is(samples);
unsigned seed = 0x9e3779b9u;
for (int t = 0; t < samples; t++) gt_spot_sample(t, &seed, &chs[t], &is[t]);
const int nth = samples < 8 ? (samples > 0 ? samples : 1) : 8;
std::vector<int> bad(nth, -1);
std::vector<std::thread> th;
for (int j = 0; j < nth; j++) {
const int t0 = (int)((long)samples * j / nth), t1 = (int)((long)samples * (j + 1) / nth);
th.emplace_back(gt_spot_check_range, t0, t1, chs.data(), is.data(), neg_r_inv, alpha_le, beta_le, gathered, &bad[j]);
}
for (auto &t : th) t.join();
for (int j = 0; j < nth; j++)
if (bad[j] >= 0) {
fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", chs[bad[j]], is[bad[j]]);
return 0;
}
return 1;
}
#endif

#if QSB_GT_HEAL








static int gt_heal(uint8_t *d_gTable, const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4],
unsigned *n_flagged, unsigned *n_rewritten) {
const unsigned cap = 65536u;
*n_flagged = 0; *n_rewritten = 0;
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *order = BN_new(),
*nri = BN_new(), *field_p = BN_new(), *alpha = BN_new(), *beta = BN_new(),
*bp = BN_new();
EC_POINT *pt = EC_POINT_new(grp);
EC_GROUP_get_order(grp, order, ctx);
EC_GROUP_get_curve_GFp(grp, field_p, NULL, NULL, ctx);
BN_lebin2bn((const uint8_t*)alpha_le, 32, alpha);
BN_lebin2bn((const uint8_t*)beta_le, 32, beta);
BN_lebin2bn(neg_r_inv, 32, nri);
uint64_t hb[4] = {0, 0, 0, 0};
BN_mod_sqr(bp, beta, field_p, ctx); BN_mul_word(bp, 7); BN_mod(bp, bp, field_p, ctx);
BN_bn2lebinpad(bp, (uint8_t*)hb, 32);
uint64_t *d_bp = NULL; unsigned *d_flags = NULL;
unsigned *h_flags = (unsigned*)malloc((size_t)(cap + 1) * sizeof(unsigned));
int rc = h_flags ? 0 : -1;
cudaError_t e = cudaSuccess;
if (rc == 0) e = cudaMalloc(&d_bp, 32);
if (rc == 0 && e == cudaSuccess) e = cudaMalloc(&d_flags, (size_t)(cap + 1) * sizeof(unsigned));
if (rc == 0 && e == cudaSuccess) e = cudaMemcpy(d_bp, hb, 32, cudaMemcpyHostToDevice);
if (rc == 0 && e == cudaSuccess) e = cudaMemset(d_flags, 0, sizeof(unsigned));
if (rc == 0 && e == cudaSuccess) {
if (!qsb_carrier_try(kernel_gt_heal_scan, QK_HEAL, dim3((GT_TOTAL_ENTRIES + 255) / 256), dim3(256),
(cudaStream_t)0, d_gTable, d_bp, d_flags, cap))
kernel_gt_heal_scan<<<(GT_TOTAL_ENTRIES + 255) / 256, 256>>>(d_gTable, d_bp, d_flags, cap);
e = cudaDeviceSynchronize();
if (e == cudaSuccess) e = cudaGetLastError();
}
if (rc == 0 && e == cudaSuccess) e = cudaMemcpy(h_flags, d_flags, sizeof(unsigned), cudaMemcpyDeviceToHost);
if (e != cudaSuccess || rc != 0) rc = -1;
else if (h_flags[0] > cap) { *n_flagged = h_flags[0]; rc = -1; }
else {
*n_flagged = h_flags[0];
if (h_flags[0] &&
cudaMemcpy(h_flags + 1, d_flags + 1, (size_t)h_flags[0] * sizeof(unsigned),
cudaMemcpyDeviceToHost) != cudaSuccess) rc = -1;
for (unsigned q = 0; rc == 0 && q < h_flags[0]; q++) {
const unsigned t = h_flags[1 + q];
int ch = -1;
for (int c = 0; c < GT_CHUNKS; c++)
if (t >= gt_offset(c) && t < gt_offset(c) + gt_entries(c)) ch = c;
if (ch < 0) { rc = -1; break; }
uint64_t want[8]; uint8_t got[64];
gt_table_scalar(k, ch, t - gt_offset(ch));
BN_mod_mul(k, k, nri, order, ctx);
EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
gt_point_to_limbs(grp, pt, x, y, alpha, beta, field_p, ctx, want);
if (cudaMemcpy(got, d_gTable + (size_t)t * 64, 64, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -1; break; }
if (memcmp(got, want, 64) != 0) {
if (cudaMemcpy(d_gTable + (size_t)t * 64, want, 64, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -1; break; }
(*n_rewritten)++;
}
}
}
cudaFree(d_bp); cudaFree(d_flags); free(h_flags);
BN_free(x); BN_free(y); BN_free(k); BN_free(order); BN_free(nri);
BN_free(field_p); BN_free(alpha); BN_free(beta); BN_free(bp);
EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
return rc;
}
#endif




static void compute_gtable(uint8_t *gTable, const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
printf("  Computing GLV12 GTable (OpenSSL fallback)...\n");
EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1); BN_CTX *ctx=BN_CTX_new();
BIGNUM *x=BN_new(),*y=BN_new(),*k=BN_new(),*stepk=BN_new(),*order=BN_new(),
*nri=BN_new(),*field_p=BN_new(),*alpha=BN_new(),*beta=BN_new();
EC_POINT *pt=EC_POINT_new(grp),*step=EC_POINT_new(grp);
EC_GROUP_get_order(grp,order,ctx); EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
BN_lebin2bn(neg_r_inv,32,nri);
for(int ch=0;ch<GT_CHUNKS;ch++) {
gt_table_scalar(k,ch,0); BN_mod_mul(k,k,nri,order,ctx);
EC_POINT_mul(grp,pt,k,NULL,NULL,ctx);
if(ch==0) BN_copy(stepk,nri);
else { BN_one(stepk); BN_lshift(stepk,stepk,gt_shift(ch));
BN_mod_mul(stepk,stepk,nri,order,ctx); }
EC_POINT_mul(grp,step,stepk,NULL,NULL,ctx);
for(unsigned d=0;d<gt_entries(ch);d++) {
uint64_t limbs[8]; gt_point_to_limbs(grp,pt,x,y,alpha,beta,field_p,ctx,limbs);
memcpy(gTable+((size_t)gt_offset(ch)+d)*64,limbs,64);
if(d+1<gt_entries(ch)) EC_POINT_add(grp,pt,pt,step,ctx);
}
}
BN_free(x);BN_free(y);BN_free(k);BN_free(stepk);BN_free(order);BN_free(nri);
BN_free(field_p);BN_free(alpha);BN_free(beta);EC_POINT_free(pt);EC_POINT_free(step);
EC_GROUP_free(grp);BN_CTX_free(ctx);
}




#if QSB_GLV11


static int qsb_s3_selfcheck(void) {
static const qsb_s3_desc_t desc[GT_GLV_TERMS]=QSB_S3_DESC_INIT;
#if QSB_Q_P18
const unsigned banks[GT_GLV_TERMS]={0,6,7,4,5,0,6,7,4,5};
#else
const unsigned banks[GT_GLV_TERMS]={0,1,2,3,4,5,0,6,7,4,5};
#endif
for(int t=0;t<GT_GLV_TERMS;t++) {
unsigned c=banks[t], w=desc[t].width;
if(desc[t].off!=gt_offset(c) || desc[t].mask!=(1u<<w)-1u) return 0;
if(c!=5 && gt_entries(c)!=(1u<<(w-(c==0?0:1)))) return 0;
}
uint64_t seed=0x243F6A8885A308D3ULL;
for(int it=0;it<4096;it++) {
uint64_t m[2][2];
for(int j=0;j<2;j++) {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][0]=seed;
seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][1]=seed%0xa2a8918ca85bafe2ULL;}
unsigned sp=it&1,sq=(it>>1)&1;qsb_s3_walker w;qsb_s3_begin(w,m[0],sp,m[1],sq);
qsb_s3_walker h;qsb_s3_begin(h,m[0],sp,m[1],sq);
for(int t=0;t<GT_GLV_TERMS;t++) {
const uint32_t got=qsb_s3_code(w,t,desc[t]);
#if QSB_Q_P18
const uint32_t want=t<QSB_S3_PSI_TERM ? q11_bigtbl_code(m[1],sq,t) : q11_bigtbl_code(m[0],sp,t-QSB_S3_PSI_TERM);
#else
const uint32_t want=t<6 ? q9_bigtbl_code(m[1],sq,t) : q11_bigtbl_code(m[0],sp,t-6);
#endif
if(got!=want) return 0;
if(t==QSB_S3_PSI_TERM) qsb_s3_psi_swap(h);
if(qsb_s3_code_half(h,t,desc[t])!=want) return 0;
}
}
#if QSB_Q_MIX



static const qsb_s3_desc_t mxf[QSB_S3_MXF_LAST+1]=QSB_S3_DESC_MXF_INIT;
static const qsb_s3_desc_t g11[11]=QSB_S3_DESC_G11;
const unsigned bank11[11]={0,1,2,3,4,5,0,6,7,4,5};
for(int t=0;t<11;t++) {
unsigned c=bank11[t], w=g11[t].width;
if(g11[t].off!=gt_offset(c) || g11[t].mask!=(1u<<w)-1u) return 0;
if(c!=5 && gt_entries(c)!=(1u<<(w-(c==0?0:1)))) return 0;
}
qsb_s3_desc_t sched[2][11]; int psi_at[2]={-1,-1}, n_at[2];
for(unsigned g=0;g<2;g++) {
int n=0;
sched[g][n++]=desc[0];
sched[g][n++]=mxf[5u*g+1u];
for(unsigned i=5u*g+2u;i<QSB_S3_MXF_LAST;i++) {
qsb_s3_desc_t d=mxf[i];
if(d.off==0u) { if(psi_at[g]>=0) return 0; i=QSB_S3_MXF_PTAIL; psi_at[g]=n; }
if(n>=10) return 0;
sched[g][n++]=d;
}
sched[g][n++]=desc[GT_GLV_TERMS-1];
n_at[g]=n;
}
if(n_at[0]!=GT_GLV_TERMS || psi_at[0]!=QSB_S3_PSI_TERM || n_at[1]!=11 || psi_at[1]!=6) return 0;
if(memcmp(&mxf[QSB_S3_MXF_LAST],&desc[GT_GLV_TERMS-1],sizeof desc[0])!=0) return 0;
if(memcmp(sched[0],desc,sizeof desc)!=0 || memcmp(sched[1],g11,sizeof g11)!=0) return 0;
for(int it=0;it<4096;it++) {
uint64_t m[2][2];
for(int j=0;j<2;j++) {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][0]=seed;
seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][1]=seed%0xa2a8918ca85bafe2ULL;
if(it<4) {m[j][0]=it&1?~0ULL:0ULL;m[j][1]=it&2?0xa2a8918ca85bafe1ULL:0ULL;}}
unsigned sp=it&1,sq=(it>>1)&1;qsb_s3_walker h;qsb_s3_begin(h,m[0],sp,m[1],sq);
for(int t=0;t<11;t++) {
const uint32_t want=t<6 ? q9_bigtbl_code(m[1],sq,t) : q11_bigtbl_code(m[0],sp,t-6);
if(t==psi_at[1]) qsb_s3_psi_swap(h);
if(qsb_s3_code_half(h,t,sched[1][t])!=want) return 0;
}
}
#if QSB_GLV_ZDEC




{
static const qsb_s3_desc_t zmx[QSB_S3_MXF_LAST+1]=QSB_S3_ZDESC_MXF_INIT;
const qsb_s3_desc_t zlast=QSB_S3_ZD(0xFFFFFFFu,170559770u,67895296u,28u);
for(unsigned i=0;i<=QSB_S3_MXF_LAST;i++)
if(zmx[i].mask!=mxf[i].mask || zmx[i].off!=mxf[i].off || zmx[i].width!=mxf[i].width ||
#if QSB_S3_BORROW_DIGIT || QSB_S3_SELECT_DIGIT || QSB_S3_ABSDIFF_DIGIT
zmx[i].centre!=(mxf[i].centre>>1)) return 0;
#elif QSB_S3_HALF_DIGIT
zmx[i].centre!=0u-(mxf[i].centre>>1)) return 0;
#else
zmx[i].centre!=1u-mxf[i].centre) return 0;
#endif
if(memcmp(&zlast,&zmx[QSB_S3_MXF_LAST],sizeof zlast)!=0) return 0;
if(QSB_ZDEC_TOPWORD!=0xFFFFFFFFu-16u*((1u<<28)-170559770u)) return 0;
for(int it=0;it<8192;it++) {


static const uint64_t ext[8][2]={{0,0},{1,0},{~0ULL,0xa2a8918ca85bafe1ULL},{0,0xa2a8918000000000ULL},
{~0ULL,0xa2a8918fffffffffULL},{~0ULL,0x0000000fffffffffULL},{0,0x0000001000000000ULL},{0x3FFFFULL,0}};
uint64_t m[2][2];
for(int j=0;j<2;j++) {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][0]=seed;
seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;m[j][1]=seed%0xa2a8918ca85bafe2ULL;
if(it<256) {m[j][0]=ext[(it>>(3*j))&7][0];m[j][1]=ext[(it>>(3*j))&7][1];}}
const unsigned sp=it<256?(it>>6)&1:it&1,sq=it<256?(it>>7)&1:(it>>1)&1;
uint64_t v[2][2]; uint32_t mm[2];
for(int j=0;j<2;j++) {
const unsigned sj=j==0?sp:sq;
uint64_t zl=m[j][0],zh=m[j][1];
if(sj) {zl=~zl;zh=~zh;zl+=1;if(zl==0)zh+=1;}
const uint32_t mj=0u-sj;
const uint64_t M=(uint64_t)mj|((uint64_t)mj<<32),H=(uint64_t)mj|((uint64_t)(mj&QSB_ZDEC_TOPWORD)<<32);
const uint64_t lo=zl+M; v[j][0]=lo; v[j][1]=zh+H+(lo<zl?1u:0u); mm[j]=mj;
}
for(unsigned g=0;g<2;g++) {
qsb_s3_walker h;qsb_s3_begin_z(h,v[0],mm[0],v[1]);
int t=0;
const qsb_s3_desc_t d0={0x3FFFFu,QSB_S3_Z0(mm[1]),0u,18u};
uint32_t got[11];
got[t++]=qsb_s3_code_z(h,d0);
got[t++]=qsb_s3_code_z(h,zmx[5u*g+1u]);
for(unsigned i=5u*g+2u;i<QSB_S3_MXF_LAST;i++) {
qsb_s3_desc_t d=zmx[i];
if(d.off==0u) {i=QSB_S3_MXF_PTAIL;qsb_s3_psi_swap_z(h);d.centre=h.signs;}
if(t>=10) return 0;
got[t++]=qsb_s3_code_z(h,d);
}
got[t++]=qsb_s3_code_z(h,zlast);
if(t!=(g?11:GT_GLV_TERMS)) return 0;
for(int u=0;u<t;u++) {
const uint32_t want=g==0 ? (u<QSB_S3_PSI_TERM ? q11_bigtbl_code(m[1],sq,u) : q11_bigtbl_code(m[0],sp,u-QSB_S3_PSI_TERM))
: (u<6 ? q9_bigtbl_code(m[1],sq,u) : q11_bigtbl_code(m[0],sp,u-6));
if(got[u]!=want) return 0;
}
#if QSB_S3_NM_MASK || QSB_S3_NM_SEED
{

qsb_s3_walker hn;qsb_s3_begin_z(hn,v[0],mm[0],v[1]);
int tn=0; uint32_t nm=0u, gn;
#define QSB_S3_NM_STEP(dd) do { gn=qsb_s3_code_zn(hn,(dd),nm); \
if((nm!=0u&&nm!=0xFFFFFFFFu)||(gn>>31)!=0u||tn>=t||(gn|(nm<<31))!=got[tn]) return 0; tn++; } while(0)
QSB_S3_NM_STEP(d0);
QSB_S3_NM_STEP(zmx[5u*g+1u]);
for(unsigned i=5u*g+2u;i<QSB_S3_MXF_LAST;i++) {
qsb_s3_desc_t d=zmx[i];
if(d.off==0u) {i=QSB_S3_MXF_PTAIL;qsb_s3_psi_swap_z(hn);d.centre=hn.signs;}
QSB_S3_NM_STEP(d);
}
QSB_S3_NM_STEP(zlast);
#undef QSB_S3_NM_STEP
if(tn!=t) return 0;
}
#endif
}
}
}
#endif
#endif
return 1;
}
#else
static int qsb_s3_selfcheck(void) {
static const qsb_s3_desc_t desc[12] = QSB_S3_DESC_INIT;
for (int t = 0; t < 12; t++) {
const int c = t % 6;
const unsigned w = c < 5 ? (unsigned)(gt_shift(c + 1) - gt_shift(c)) : 28u;
if (desc[t].off != gt_offset(c) || desc[t].width != w || desc[t].mask != (1u << w) - 1u) return 0;
if (desc[t].centre != (c == 0 ? 0u : c == 5 ? 170559770u : 1u << w)) return 0;
if (c < 5 && gt_entries(c) != (c == 0 ? 1u << 18 : 1u << (w - 1))) return 0;
}
if (gt_entries(5) != 85279885u || ((0xa2a8918ca85bafe2ULL >> 36) | 1ULL) != 170559769ULL) return 0;
uint64_t s = 0x243F6A8885A308D3ULL;
for (int it = 0; it < 4096; it++) {
uint64_t m[2][2];
for (int j = 0; j < 2; j++) {
s ^= s << 13; s ^= s >> 7; s ^= s << 17; m[j][0] = s;
s ^= s << 13; s ^= s >> 7; s ^= s << 17; m[j][1] = s % 0xa2a8918ca85bafe2ULL;
if (it < 4) { m[j][0] = it & 1 ? ~0ULL : 0ULL; m[j][1] = it & 2 ? 0xa2a8918ca85bafe1ULL : 0ULL; }
}
const unsigned sp = it & 1, sq = (it >> 1) & 1;
qsb_s3_walker w; qsb_s3_begin(w, m[0], sp, m[1], sq);
qsb_s3_walker h; qsb_s3_begin(h, m[0], sp, m[1], sq);
for (int t = 0; t < 12; t++) {
const uint32_t got = qsb_s3_code(w, t, desc[t]);
const uint32_t want = t < 6 ? q9_bigtbl_code(m[1], sq, t) : q9_bigtbl_code(m[0], sp, t - 6);
if (got != want) return 0;
if (t == QSB_S3_PSI_TERM) qsb_s3_psi_swap(h);
if (qsb_s3_code_half(h, t, desc[t]) != want) return 0;
}
}
return 1;
}
#endif
#else









static void gt_build_ladders(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(),
*order = BN_new(), *nri = BN_new(), *bscal = BN_new(), *field_p=BN_new(),
*alpha=BN_new(), *beta=BN_new();
EC_POINT *base = EC_POINT_new(grp), *step = EC_POINT_new(grp), *acc = EC_POINT_new(grp);

EC_GROUP_get_order(grp, order, ctx);
BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
BN_lebin2bn(neg_r_inv, 32, nri);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
#if QSB_TABLE_BASE_A
BN_copy(bscal, nri);
#else
BN_mod_mul(bscal, inv2, nri, order, ctx);
#endif
EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
memset(hL, 0, (size_t)GT_CHUNKS * GT_LO * 8 * sizeof(uint64_t));
memset(hH, 0, (size_t)GT_CHUNKS * GT_HI * 8 * sizeof(uint64_t));
#if QSB_STARTUP_TRIM
EC_POINT *pts[GT_HI];
for (int i = 0; i < GT_HI; i++) pts[i] = EC_POINT_new(grp);
#endif
for (int ch = 0; ch < GT_CHUNKS; ch++) {
if (ch > 0) { BN_set_word(shift, 1ul << (gt_shift(ch) - gt_shift(ch-1))); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
EC_POINT_copy(acc, base);
#if QSB_STARTUP_TRIM
for (int lo = 1; lo < GT_LO; lo++) {
EC_POINT_copy(pts[lo], acc);
EC_POINT_add(grp, acc, acc, base, ctx);
}
if (!EC_POINTs_make_affine(grp, (size_t)(GT_LO - 1), pts + 1, ctx)) {
fprintf(stderr, "ERROR: GTable low ladder affine conversion failed\n"); exit(1);
}
for (int lo = 1; lo < GT_LO; lo++)
gt_point_to_limbs(grp, pts[lo], x, y, alpha,beta,field_p,ctx,
hL + ((size_t)ch * GT_LO + lo) * 8);
#else
for (int lo = 1; lo < GT_LO; lo++) {
gt_point_to_limbs(grp, acc, x, y, alpha,beta,field_p,ctx,
hL + ((size_t)ch * GT_LO + lo) * 8);
EC_POINT_add(grp, acc, acc, base, ctx);
}
#endif
BN_set_word(shift, 256);
EC_POINT_mul(grp, step, NULL, base, shift, ctx);
EC_POINT_copy(acc, step);
#if QSB_STARTUP_TRIM
{
const int n_hi = (int)(gt_entries(ch) >> 7);
for (int hi = 1; hi < n_hi; hi++) {
EC_POINT_copy(pts[hi], acc);
EC_POINT_add(grp, acc, acc, step, ctx);
}
if (!EC_POINTs_make_affine(grp, (size_t)(n_hi - 1), pts + 1, ctx)) {
fprintf(stderr, "ERROR: GTable high ladder affine conversion failed\n"); exit(1);
}
for (int hi = 1; hi < n_hi; hi++)
gt_point_to_limbs(grp, pts[hi], x, y, alpha,beta,field_p,ctx,
hH + ((size_t)ch * GT_HI + hi) * 8);
}
#else
for (int hi = 1; hi < (int)(gt_entries(ch) >> 7); hi++) {
gt_point_to_limbs(grp, acc, x, y, alpha,beta,field_p,ctx,
hH + ((size_t)ch * GT_HI + hi) * 8);
EC_POINT_add(grp, acc, acc, step, ctx);
}
#endif
}
#if QSB_STARTUP_TRIM
for (int i = 0; i < GT_HI; i++) EC_POINT_free(pts[i]);
#endif
BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
BN_free(nri); BN_free(bscal); BN_free(field_p); BN_free(alpha); BN_free(beta);
EC_POINT_free(base); EC_POINT_free(step); EC_POINT_free(acc);
EC_GROUP_free(grp); BN_CTX_free(ctx);
}






static void gt_spot_sample(int t, unsigned *seed, int *ch_out, int *i_out) {
int ch, i;
if (t < GT_CHUNKS * 4) {
ch = t / 4;
const int corner[4] = {0, 1, 2, (int)gt_entries(ch) - 1};
i = corner[t % 4];
} else {
*seed = *seed * 1664525u + 1013904223u;
ch = (int)(*seed >> 28) % GT_CHUNKS;
i = (int)((*seed >> 4) & (gt_entries(ch) - 1));
}
*ch_out = ch; *i_out = i;
}
static int gt_spot_check(const uint8_t *gTable, int samples,
const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4],
const uint8_t *gathered) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *inv2 = BN_new(), *order = BN_new(),
*nri = BN_new(), *half_nri = BN_new(), *field_p=BN_new(),
*alpha=BN_new(), *beta=BN_new();
EC_POINT *pt = EC_POINT_new(grp);
uint64_t want[8];
int ok = 1;
unsigned seed = 0x9e3779b9u;
EC_GROUP_get_order(grp, order, ctx);
BN_set_word(k, 2); BN_mod_inverse(inv2, k, order, ctx);
BN_lebin2bn(neg_r_inv, 32, nri);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
#if QSB_TABLE_BASE_A
BN_copy(half_nri, nri);
#else
BN_mod_mul(half_nri, inv2, nri, order, ctx);
#endif
for (int t = 0; t < samples && ok; t++) {

int ch, i;
gt_spot_sample(t, &seed, &ch, &i);

BN_one(k);
BN_lshift(k, k, gt_shift(ch));
BN_mul_word(k, (BN_ULONG)(2*i + 1));
BN_mod_mul(k, k, half_nri, order, ctx);
EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
gt_point_to_limbs(grp, pt, x, y, alpha,beta,field_p,ctx,want);
const uint8_t *rec = gathered ? gathered + (size_t)t * 64
: gTable + ((size_t)gt_offset(ch) + i) * 64;
if (memcmp(rec, want, 32) != 0 ||
memcmp(rec + 32, want + 4, 32) != 0) {
fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
ok = 0;
}
}
BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
BN_free(field_p); BN_free(alpha); BN_free(beta);
EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
return ok;
}




#ifndef QSB_BATCH_AFFINE_FALLBACK
#define QSB_BATCH_AFFINE_FALLBACK 1
#endif
#define QSB_FALLBACK_AFFINE_BATCH 8192
static void compute_gtable(uint8_t *gTable, const uint8_t neg_r_inv[32],
const uint64_t alpha_le[4], const uint64_t beta_le[4]) {

printf("  Computing GTable (OpenSSL fallback)...\n");
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx = BN_CTX_new();
BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(), *order = BN_new(),
*nri = BN_new(), *bscal = BN_new(), *field_p=BN_new(),
*alpha=BN_new(), *beta=BN_new();
EC_POINT *base = EC_POINT_new(grp), *pt = EC_POINT_new(grp), *two_base = EC_POINT_new(grp);

EC_GROUP_get_order(grp, order, ctx);
BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
BN_lebin2bn(neg_r_inv, 32, nri);
EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
BN_lebin2bn((const uint8_t*)beta_le,32,beta);
#if QSB_TABLE_BASE_A
BN_copy(bscal, nri);
#else
BN_mod_mul(bscal, inv2, nri, order, ctx);
#endif
EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
#if QSB_BATCH_AFFINE_FALLBACK



EC_POINT *batch[QSB_FALLBACK_AFFINE_BATCH];
for (int j = 0; j < QSB_FALLBACK_AFFINE_BATCH; j++) {
batch[j] = EC_POINT_new(grp);
if (!batch[j]) { fprintf(stderr, "OOM: GTable affine batch\n"); exit(1); }
}
#endif
for (int ch = 0; ch < GT_CHUNKS; ch++) {
if (ch > 0) { BN_set_word(shift, 1ul << (gt_shift(ch) - gt_shift(ch-1))); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
BN_set_word(shift, 2); EC_POINT_mul(grp, two_base, NULL, base, shift, ctx);
EC_POINT_copy(pt, base);
#if QSB_BATCH_AFFINE_FALLBACK
for (unsigned start = 0; start < gt_entries(ch); start += QSB_FALLBACK_AFFINE_BATCH) {
unsigned count = gt_entries(ch) - start;
if (count > QSB_FALLBACK_AFFINE_BATCH) count = QSB_FALLBACK_AFFINE_BATCH;
for (unsigned j = 0; j < count; j++) {
if (!EC_POINT_copy(batch[j], pt)) {
fprintf(stderr, "ERROR: GTable affine point copy failed\n"); exit(1);
}
if (start + j + 1 < gt_entries(ch) &&
!EC_POINT_add(grp, pt, pt, two_base, ctx)) {
fprintf(stderr, "ERROR: GTable affine point step failed\n"); exit(1);
}
}
if (!EC_POINTs_make_affine(grp, count, batch, ctx)) {
fprintf(stderr, "ERROR: GTable batch affine conversion failed\n"); exit(1);
}
for (unsigned j = 0; j < count; j++) {
uint64_t limbs[8];
gt_point_to_limbs(grp, batch[j], x, y, alpha, beta, field_p, ctx, limbs);
size_t off = ((size_t)gt_offset(ch) + start + j) * 64;
memcpy(gTable + off, limbs, sizeof(limbs));
}
}
#else
for (unsigned d = 0; d < gt_entries(ch); d++) {
EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
BN_mod_mul(x,x,alpha,field_p,ctx);
BN_mod_mul(y,y,beta, field_p,ctx);
uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
size_t off = ((size_t)gt_offset(ch) + d) * 64;
memcpy(gTable + off, xb, 32);
memcpy(gTable + off + 32, yb, 32);
if (d < gt_entries(ch) - 1) EC_POINT_add(grp, pt, pt, two_base, ctx);
}
#endif
}
#if QSB_BATCH_AFFINE_FALLBACK
for (int j = 0; j < QSB_FALLBACK_AFFINE_BATCH; j++) EC_POINT_free(batch[j]);
#endif
BN_free(x);BN_free(y);BN_free(shift);BN_free(inv2);BN_free(order);BN_free(nri);BN_free(bscal);
BN_free(field_p);BN_free(alpha);BN_free(beta);
EC_POINT_free(base);EC_POINT_free(pt);EC_POINT_free(two_base);
EC_GROUP_free(grp);BN_CTX_free(ctx);
}

#endif


typedef struct {
uint32_t n, t;
uint32_t total_preimage_len;
uint32_t tail_section_len;
uint32_t tx_suffix_len;
uint32_t prefix_remainder_len;
uint32_t midstate[8];
uint8_t *prefix_remainder;
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

typedef struct {
uint64_t alpha[4];
uint64_t beta[4];
uint64_t invu[4];
uint64_t u2r_iso[8];
uint32_t xneg;
} qsb_iso_params_t;




static int qsb_make_iso_params(const digest_params_t *dp,qsb_iso_params_t *out){
static const uint8_t p_be[32]={
0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F};
BN_CTX *ctx=BN_CTX_new();
BIGNUM *p=BN_new(),*x=BN_new(),*y=BN_new(),*alpha=BN_new(),*u=BN_new(),
*exp=BN_new(),*check=BN_new(),*beta=BN_new(),*invu=BN_new(),
*xt=BN_new(),*yt=BN_new();
int ok=ctx&&p&&x&&y&&alpha&&u&&exp&&check&&beta&&invu&&xt&&yt;
if(ok)ok=BN_bin2bn(p_be,32,p)!=NULL && BN_lebin2bn(dp->u2r_x,32,x)!=NULL
&& BN_lebin2bn(dp->u2r_y,32,y)!=NULL;
if(ok)ok=BN_mod_inverse(alpha,x,p,ctx)!=NULL;
if(ok){
BN_copy(exp,p);BN_add_word(exp,1);BN_rshift(exp,exp,2);
BN_mod_exp(u,alpha,exp,p,ctx);BN_mod_sqr(check,u,p,ctx);
out->xneg=(BN_cmp(check,alpha)!=0);
if(out->xneg){BN_mod_sub(alpha,p,alpha,p,ctx);BN_mod_exp(u,alpha,exp,p,ctx);}
BN_mod_sqr(check,u,p,ctx);
ok=BN_cmp(check,alpha)==0;
}
if(ok){
BN_mod_mul(beta,alpha,u,p,ctx);
ok=BN_mod_inverse(invu,u,p,ctx)!=NULL;
}
if(ok){
BN_one(xt);if(out->xneg)BN_sub(xt,p,xt);
BN_mod_mul(yt,beta,y,p,ctx);
ok=BN_bn2lebinpad(alpha,(uint8_t*)out->alpha,32)==32
&& BN_bn2lebinpad(beta,(uint8_t*)out->beta,32)==32
&& BN_bn2lebinpad(invu,(uint8_t*)out->invu,32)==32
&& BN_bn2lebinpad(xt,(uint8_t*)out->u2r_iso,32)==32
&& BN_bn2lebinpad(yt,(uint8_t*)(out->u2r_iso+4),32)==32;
}
BN_free(p);BN_free(x);BN_free(y);BN_free(alpha);BN_free(u);BN_free(exp);
BN_free(check);BN_free(beta);BN_free(invu);BN_free(xt);BN_free(yt);BN_CTX_free(ctx);
if(!ok)fprintf(stderr,"ERROR: isomorphic coordinate setup failed\n");
return ok?0:-1;
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


static void qsb_host_unrank(uint64_t rank, int n, int t, uint8_t *out) {
int lo = 0;
for (int i = 0; i < t; i++) {
int c = lo;
for (;;) { uint64_t cnt = binom_u64(n - c - 1, t - i - 1); if (rank < cnt) break; rank -= cnt; c++; }
out[i] = (uint8_t)c; lo = c + 1;
}
}
static uint64_t qsb_host_rank(const uint8_t *c, int k, int n) {
uint64_t r = 0; int prev = -1;
for (int i = 0; i < k; i++) { for (int j = prev + 1; j < c[i]; j++) r += binom_u64(n - j - 1, k - i - 1); prev = c[i]; }
return r;
}

static double binom_d(int n, int k) {
if (k < 0 || n < 0 || k > n) return 0.0;
if (k > n - k) k = n - k;
double r = 1.0;
for (int i = 0; i < k; i++) r = r * (double)(n - i) / (double)(i + 1);
return r;
}


static void unrank_combo_host(uint64_t rank, int n, int t, uint8_t *out) {
int lo = 0;
for (int i = 0; i < t; i++) {
int k = t - i - 1;
for (;;) {
uint64_t c = binom_u64(n - lo - 1, k);
if (rank < c) break;
rank -= c; lo++;
}
out[i] = (uint8_t)lo; lo++;
}
}

#if QSB_HOST_VERIFY
static uint8_t g_hv_win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
#include "qsb_host_verify.h"
#if QSB_HOST_PRODUCERS && QSB_SLOT_PIPELINE && ZLAB_HITPATH
#include "host_producers.h"
#define QSB_HP_ON 1
#if (QSB_FIRST_LOCAL || QSB_FIRST_DEVICE) && !QSB_HP_V3
#error "Consumer-local first requires host producer v3"
#endif
#endif





#ifndef QSB_HOST_BLOCKING
#define QSB_HOST_BLOCKING 1
#endif

#ifndef QSB_CPU_GRIND
#define QSB_CPU_GRIND 1
#endif
#if QSB_CPU_GRIND
#include "../../CpuGrindSubset.h"
#endif
#endif

#if QSB_HP_FIRST_ONLY_UPLOAD
#if !(QSB_HOST_PRODUCERS && QSB_SLOT_PIPELINE && QSB_HOST_VERIFY && ZLAB_HITPATH && QSB_HP_V3 && QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA && QSB_HIT_NO_COMBO)
#error "first-only host upload requires checked v3 producers and the paired tag-only digest"
#endif
#endif





static void build_epoch_prefix(const digest_params_t *dp, int window_start, int s_early,
const uint8_t *early, uint8_t *scratch,
uint32_t mid_out[8], uint8_t *rem_out, int *rem_len_out) {
size_t pos = 0;
for (uint32_t i = 0; i < dp->prefix_remainder_len; i++) scratch[pos++] = dp->prefix_remainder[i];
int sel = 0;
for (int i = 0; i < window_start; i++) {
if (sel < s_early && (int)early[sel] == i) { sel++; continue; }
memcpy(scratch + pos, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE);
pos += SIG_PUSH_SIZE;
}
SHA256_CTX ctx;
SHA256_Init(&ctx);
for (int i = 0; i < 8; ++i) ctx.h[i] = dp->midstate[i];
size_t blocks = pos / 64;
for (size_t i = 0; i < blocks; ++i) SHA256_Transform(&ctx, scratch + i * 64);
for (int i = 0; i < 8; ++i) mid_out[i] = ctx.h[i];
*rem_len_out = (int)(pos - blocks * 64);
if (*rem_len_out > 0) memcpy(rem_out, scratch + blocks * 64, (size_t)*rem_len_out);
}























#ifndef QSB_NO_SUMMARY_FSYNC
#define QSB_NO_SUMMARY_FSYNC 1
#endif
#ifndef QSB_LAZY_MODULES
#define QSB_LAZY_MODULES 1
#endif
#ifndef QSB_FAST_TEARDOWN
#define QSB_FAST_TEARDOWN 1
#endif
#ifndef QSB_FAST_TEARDOWN_PIN
#define QSB_FAST_TEARDOWN_PIN 1
#endif
#if QSB_FAST_TEARDOWN
#include <atomic>
#include <thread>
#include <semaphore.h>
static sem_t g_ft_sem;
static volatile sig_atomic_t g_ft_sem_ok = 0;
#endif
#if QSB_NO_SUMMARY_FSYNC
#define QSB_SUMMARY_FSYNC(f) ((void)(f))
#else
#define QSB_SUMMARY_FSYNC(f) fsync(fileno(f))
#endif






static volatile FILE *g_summary_f = NULL;
static volatile uint64_t g_hit_counter = 0;
static volatile uint64_t g_total_searched = 0;
#if QSB_SLOT_PIPELINE




static volatile sig_atomic_t g_stop_signal = 0;
static volatile sig_atomic_t g_stop_polled = 0;
#endif

static void on_term_signal(int sig) {
#if QSB_SLOT_PIPELINE
#if QSB_FAST_TEARDOWN
if (g_stop_polled) { g_stop_signal = sig; if (g_ft_sem_ok) sem_post(&g_ft_sem); return; }
#else
if (g_stop_polled) { g_stop_signal = sig; return; }
#endif
#endif
if (g_summary_f) {
time_t now_epoch = time(NULL);
fprintf((FILE*)g_summary_f,
"STATUS=KILLED %ld signal=%d total_attempts=%llu hits=%llu\n",
(long)now_epoch, sig,
(unsigned long long)g_total_searched,
(unsigned long long)g_hit_counter);
fflush((FILE*)g_summary_f);
QSB_SUMMARY_FSYNC((FILE*)g_summary_f);
}

signal(sig, SIG_DFL);
raise(sig);
}

#if QSB_TABLE_L2_WINDOW





static void qsb_table_l2_window(cudaStream_t *streams, int n_streams,
const uint8_t *d_gt, size_t gt_sz) {
int dev = 0, max_persist = 0, max_window = 0;
size_t limit = 0, want = 0;
int applied = 0;
cudaError_t we = cudaGetDevice(&dev);
if (we == cudaSuccess) we = cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
if (we == cudaSuccess) we = cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
if (we == cudaSuccess && max_persist > 0 && max_window > 0) {
we = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)max_persist);
if (we == cudaSuccess) we = cudaDeviceGetLimit(&limit, cudaLimitPersistingL2CacheSize);
if (we == cudaSuccess) {
want = gt_sz;
#if QSB_S3


if (want > (size_t)GT_DENSE_ENTRIES * 64u) want = (size_t)GT_DENSE_ENTRIES * 64u;
#endif
if (want > limit) want = limit;
if (want > (size_t)max_window) want = (size_t)max_window;
}
if (we == cudaSuccess && want > 0) {
cudaStreamAttrValue av;
memset(&av, 0, sizeof(av));
av.accessPolicyWindow.base_ptr = (void *)d_gt;
av.accessPolicyWindow.num_bytes = want;
av.accessPolicyWindow.hitRatio = 1.0f;
av.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
av.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
for (int s = 0; s < n_streams && we == cudaSuccess; s++)
we = cudaStreamSetAttribute(streams[s], cudaStreamAttributeAccessPolicyWindow, &av);
applied = (we == cudaSuccess);
}
}
if (applied)
printf("  Table L2 window: %.1f of %.1f MiB persisting on %d stream(s) (limit %.1f MiB, max window %.1f MiB)\n",
(double)want / 1048576.0, (double)gt_sz / 1048576.0, n_streams,
(double)limit / 1048576.0, (double)max_window / 1048576.0);
else
printf("  Table L2 window not applied (%s)\n",
we == cudaSuccess ? "no persisting L2 on this device" : cudaGetErrorString(we));
(void)cudaGetLastError();
fflush(stdout);
}
#endif








#if QSB_TREE_TOP_SHFL
#define QSB_CARRIER_TREE_TOP_SHFL QSB_CARRIER_KV(QSB_TREE_TOP_SHFL)
#else
#define QSB_CARRIER_TREE_TOP_SHFL ""
#endif
#if QSB_CTA_WINDOW_SPLIT
#define QSB_CARRIER_CTA_WINDOW_SPLIT QSB_CARRIER_KV(QSB_CTA_WINDOW_SPLIT)
#else
#define QSB_CARRIER_CTA_WINDOW_SPLIT ""
#endif
#if QSB_ROOT_LUT32
#define QSB_CARRIER_ROOT_LUT32 QSB_CARRIER_KV(QSB_ROOT_LUT32)
#else
#define QSB_CARRIER_ROOT_LUT32 ""
#endif
#if QSB_ROOT_LUT40
#define QSB_CARRIER_ROOT_LUT40 QSB_CARRIER_KV(QSB_ROOT_LUT40) QSB_CARRIER_KV(QSB_ROOT_LUT40_SHORT)
#else
#define QSB_CARRIER_ROOT_LUT40 ""
#endif
#if QSB_FIRST_FLAT_FAST
#define QSB_CARRIER_FIRST_FLAT_FAST QSB_CARRIER_KV(QSB_FIRST_FLAT_FAST)
#else
#define QSB_CARRIER_FIRST_FLAT_FAST ""
#endif
#if QSB_OUTER_PAIR
#define QSB_CARRIER_OUTER_PAIR QSB_CARRIER_KV(QSB_OUTER_PAIR)
#else
#define QSB_CARRIER_OUTER_PAIR ""
#endif
#if QSB_ROOT_LUT_GLOBAL
#define QSB_CARRIER_ROOT_LUT_GLOBAL QSB_CARRIER_KV(QSB_ROOT_LUT_GLOBAL)
#else
#define QSB_CARRIER_ROOT_LUT_GLOBAL ""
#endif
#if QSB_ROOT_PARK_B
#define QSB_CARRIER_ROOT_PARK_B QSB_CARRIER_KV(QSB_ROOT_PARK_B)
#else
#define QSB_CARRIER_ROOT_PARK_B ""
#endif
#if QSB_DEN_CROSS_IDLE
#define QSB_CARRIER_DEN_CROSS_IDLE QSB_CARRIER_KV(QSB_DEN_CROSS_IDLE)
#else
#define QSB_CARRIER_DEN_CROSS_IDLE ""
#endif
#if QSB_DEN_CROSS_PRE
#define QSB_CARRIER_DEN_CROSS_PRE QSB_CARRIER_KV(QSB_DEN_CROSS_PRE)
#else
#define QSB_CARRIER_DEN_CROSS_PRE ""
#endif
#if QSB_K2S_CENTER_SQR
#define QSB_CARRIER_CENTER_SQR QSB_CARRIER_KV(QSB_K2S_CENTER_SQR)
#else
#define QSB_CARRIER_CENTER_SQR ""
#endif
#if QSB_K2S_SUM_BASIS
#define QSB_CARRIER_SUM_BASIS QSB_CARRIER_KV(QSB_K2S_SUM_BASIS)
#else
#define QSB_CARRIER_SUM_BASIS ""
#endif
#if QSB_FIRST_DEVICE
#define QSB_CARRIER_FIRST_DEVICE "fd1"
#else
#define QSB_CARRIER_FIRST_DEVICE ""
#endif
#if QSB_FIRST_LOCAL
#define QSB_CARRIER_FIRST_LOCAL "fl1"
#else
#define QSB_CARRIER_FIRST_LOCAL ""
#endif
#if QSB_FIRST_GWK7
#define QSB_CARRIER_FIRST_GWK7 "fg7"
#else
#define QSB_CARRIER_FIRST_GWK7 ""
#endif
#if QSB_FIRST_PLANES
#define QSB_CARRIER_FIRST_PLANES "fp2"
#else
#define QSB_CARRIER_FIRST_PLANES ""
#endif
#if QSB_FIRST_PACK8
#define QSB_CARRIER_FIRST_PACK QSB_CARRIER_KV(QSB_FIRST_SLOTS)
#else
#define QSB_CARRIER_FIRST_PACK ""
#endif
#if QSB_K2S_CENTER_FUSED
#define QSB_CARRIER_CENTER_FUSED QSB_CARRIER_KV(QSB_K2S_CENTER_FUSED)
#else
#define QSB_CARRIER_CENTER_FUSED ""
#endif
#if QSB_S3_FINAL_PREFETCH
#define QSB_CARRIER_FINAL_PREFETCH QSB_CARRIER_KV(QSB_S3_FINAL_PREFETCH)
#else
#define QSB_CARRIER_FINAL_PREFETCH ""
#endif
#if QSB_SHA_LEA_GATE
#define QSB_CARRIER_LEA_GATE QSB_CARRIER_KV(QSB_SHA_LEA_GATE)
#else
#define QSB_CARRIER_LEA_GATE ""
#endif
#if QSB_YP_LEGACY_DC
#define QSB_CARRIER_LEGACY_DC QSB_CARRIER_KV(QSB_YP_LEGACY_DC)
#else
#define QSB_CARRIER_LEGACY_DC ""
#endif
#if QSB_YP_MAC_PORT
#define QSB_CARRIER_MAC_PORT QSB_CARRIER_KV(QSB_YP_MAC_PORT) QSB_CARRIER_KV(QSB_YP_MAC) QSB_CARRIER_KV(QSB_YP_DC)
#else
#define QSB_CARRIER_MAC_PORT ""
#endif
#if QSB_TREE_LIVE_MASK
#define QSB_CARRIER_TREE_LIVE_MASK QSB_CARRIER_KV(QSB_TREE_LIVE_MASK)
#else
#define QSB_CARRIER_TREE_LIVE_MASK ""
#endif
#if QSB_FOLD_REG
#define QSB_CARRIER_FOLD_REG QSB_CARRIER_KV(QSB_FOLD_REG)
#else
#define QSB_CARRIER_FOLD_REG ""
#endif
#if QSB_TREE_ROW128
#define QSB_CARRIER_TREE_ROW128 QSB_CARRIER_KV(QSB_TREE_ROW128)
#else
#define QSB_CARRIER_TREE_ROW128 ""
#endif
#if QSB_POOL_RCONST
#define QSB_CARRIER_POOL_RCONST QSB_CARRIER_KV(QSB_POOL_RCONST)
#else
#define QSB_CARRIER_POOL_RCONST ""
#endif
#if QSB_ZZ3_LATE
#define QSB_CARRIER_ZZ3_LATE QSB_CARRIER_KV(QSB_ZZ3_LATE)
#else
#define QSB_CARRIER_ZZ3_LATE ""
#endif
#if QSB_EC_PSI_ZZ
#define QSB_CARRIER_EC_PSI_ZZ QSB_CARRIER_KV(QSB_EC_PSI_ZZ)
#else
#define QSB_CARRIER_EC_PSI_ZZ ""
#endif
#if QSB_YOFF_S
#define QSB_CARRIER_YOFF_S QSB_CARRIER_KV(QSB_YOFF_S)
#else
#define QSB_CARRIER_YOFF_S ""
#endif
#if QSB_PW_HI_APPROX
#define QSB_CARRIER_PW_HI_APPROX QSB_CARRIER_KV(QSB_PW_HI_APPROX)
#else
#define QSB_CARRIER_PW_HI_APPROX ""
#endif
#if QSB_TREE_KARATSUBA
#define QSB_CARRIER_TREE_KARATSUBA QSB_CARRIER_KV(QSB_TREE_KARATSUBA)
#else
#define QSB_CARRIER_TREE_KARATSUBA ""
#endif
#if QSB_INVERSE_PACK_CARRY
#define QSB_CARRIER_INVERSE_PACK_CARRY QSB_CARRIER_KV(QSB_INVERSE_PACK_CARRY)
#else
#define QSB_CARRIER_INVERSE_PACK_CARRY ""
#endif
#if QSB_GT_PREFIX_CACHE
#define QSB_CARRIER_GT_PREFIX_CACHE QSB_CARRIER_KV(QSB_GT_PREFIX_CACHE)
#else
#define QSB_CARRIER_GT_PREFIX_CACHE ""
#endif
#if QSB_PW_SPLIT_ACC
#define QSB_CARRIER_PW_SPLIT_ACC QSB_CARRIER_KV(QSB_PW_SPLIT_ACC)
#else
#define QSB_CARRIER_PW_SPLIT_ACC ""
#endif
#if QSB_PW_PAIR
#define QSB_CARRIER_PW_PAIR QSB_CARRIER_KV(QSB_PW_PAIR)
#else
#define QSB_CARRIER_PW_PAIR ""
#endif
#if QSB_PW_BIT32
#define QSB_CARRIER_PW_BIT32 QSB_CARRIER_KV(QSB_PW_BIT32)
#else
#define QSB_CARRIER_PW_BIT32 ""
#endif
#if QSB_S3_HALF_DIGIT
#define QSB_CARRIER_HALF_DIGIT "hd1"
#else
#define QSB_CARRIER_HALF_DIGIT ""
#endif
#if QSB_GATHER_ALL_FETCH64 == 2
#define QSB_CARRIER_ALL_FETCH64 "cf64"
#elif QSB_GATHER_ALL_FETCH64 == 1
#define QSB_CARRIER_ALL_FETCH64 "af64"
#else
#define QSB_CARRIER_ALL_FETCH64 ""
#endif
#if QSB_S3_ABSDIFF_DIGIT == 3
#define QSB_CARRIER_BORROW_DIGIT "ad3"
#elif QSB_S3_ABSDIFF_DIGIT == 2
#define QSB_CARRIER_BORROW_DIGIT "ad2"
#elif QSB_S3_ABSDIFF_DIGIT == 1
#define QSB_CARRIER_BORROW_DIGIT "ad1"
#elif QSB_S3_SELECT_DIGIT
#define QSB_CARRIER_BORROW_DIGIT "sd1"
#elif QSB_S3_BORROW_DIGIT
#define QSB_CARRIER_BORROW_DIGIT "bd1"
#else
#define QSB_CARRIER_BORROW_DIGIT ""
#endif
#if QSB_GATE_REC_ROLL
#define QSB_CARRIER_REC_ROLL "gr1;"
#else
#define QSB_CARRIER_REC_ROLL ""
#endif

#if QSB_GATE_REC_CALL
#define QSB_CARRIER_REC_CALL "gc1;"
#else
#define QSB_CARRIER_REC_CALL ""
#endif
#if QSB_OUTER_FRONT_CALL
#define QSB_CARRIER_OUTER_FRONT_CALL "ofc1;"
#else
#define QSB_CARRIER_OUTER_FRONT_CALL ""
#endif
#if QSB_FRONT3_WORDS_ONLY
#define QSB_CARRIER_FRONT3_WORDS_ONLY "fw1;"
#else
#define QSB_CARRIER_FRONT3_WORDS_ONLY ""
#endif
#if QSB_FRONT3_PUBLISH_A
#define QSB_CARRIER_FRONT3_PUBLISH_A "fpa1;"
#else
#define QSB_CARRIER_FRONT3_PUBLISH_A ""
#endif
#if QSB_FRONT3_PUBLISH_AB == 2
#define QSB_CARRIER_FRONT3_PUBLISH_AB "fpab2;"
#elif QSB_FRONT3_PUBLISH_AB
#define QSB_CARRIER_FRONT3_PUBLISH_AB "fpab1;"
#else
#define QSB_CARRIER_FRONT3_PUBLISH_AB ""
#endif
#if QSB_TREE_BOTTOM_FUSE
#define QSB_CARRIER_TREE_BOTTOM_FUSE "tbf1;"
#else
#define QSB_CARRIER_TREE_BOTTOM_FUSE ""
#endif
#if QSB_TREE_BOTTOM_SHFL
#define QSB_CARRIER_TREE_BOTTOM_SHFL "tbs1;"
#else
#define QSB_CARRIER_TREE_BOTTOM_SHFL ""
#endif
#if QSB_TREE_LEAF_SHFL
#define QSB_CARRIER_TREE_LEAF_SHFL "tls1;"
#else
#define QSB_CARRIER_TREE_LEAF_SHFL ""
#endif
#define QSB_CARRIER_KNOBS QSB_CARRIER_TREE_LEAF_SHFL QSB_CARRIER_TREE_BOTTOM_SHFL QSB_CARRIER_TREE_BOTTOM_FUSE QSB_CARRIER_FRONT3_PUBLISH_AB QSB_CARRIER_FRONT3_PUBLISH_A QSB_CARRIER_FRONT3_WORDS_ONLY QSB_CARRIER_OUTER_FRONT_CALL QSB_CARRIER_REC_CALL QSB_CARRIER_REC_ROLL QSB_CARRIER_BORROW_DIGIT QSB_CARRIER_ALL_FETCH64 QSB_CARRIER_HALF_DIGIT QSB_CARRIER_PW_BIT32 QSB_CARRIER_PW_PAIR QSB_CARRIER_PW_SPLIT_ACC QSB_CARRIER_YOFF_S QSB_CARRIER_ZZ3_LATE QSB_CARRIER_EC_PSI_ZZ QSB_CARRIER_POOL_RCONST QSB_CARRIER_TREE_ROW128 QSB_CARRIER_FOLD_REG QSB_CARRIER_TREE_LIVE_MASK QSB_CARRIER_MAC_PORT QSB_CARRIER_LEGACY_DC QSB_CARRIER_LEA_GATE QSB_LEA_KNOBS QSB_CARRIER_FINAL_PREFETCH QSB_CARRIER_CENTER_FUSED QSB_CARRIER_FIRST_GWK7 QSB_CARRIER_FIRST_PLANES QSB_CARRIER_FIRST_PACK QSB_CARRIER_SUM_BASIS QSB_CARRIER_CENTER_SQR QSB_CARRIER_KV(QSB_ZEROS_N) QSB_CARRIER_KV(QSB_S3) \
QSB_CARRIER_KV(QSB_SE_WINDOWS) QSB_CARRIER_KV(QSB_SE_BLOCK) QSB_CARRIER_KV(MAX_T) \
QSB_CARRIER_KV(QSB_950_PACK) QSB_CARRIER_KV(QSB_BATCH_AFFINE_FALLBACK) QSB_CARRIER_KV(QSB_BIGTBL) \
QSB_CARRIER_KV(QSB_CHAIN_ANCHOR_UPDATE) QSB_CARRIER_KV(QSB_CHAIN_MUL_LEAN) \
QSB_CARRIER_KV(QSB_CHAIN_UNROLL) QSB_CARRIER_KV(QSB_DIGIT_SHIFT) QSB_CARRIER_KV(QSB_EPOCH_FAST) \
QSB_CARRIER_KV(QSB_EPOCH_GROUPS) QSB_CARRIER_KV(QSB_FINAL_CARRY) QSB_CARRIER_KV(QSB_FUSE_X3) \
QSB_CARRIER_KV(QSB_FX3_SIGNED) QSB_CARRIER_KV(QSB_FX3_SPLIT3P) QSB_CARRIER_KV(QSB_FX3_Z9) \
QSB_CARRIER_KV(QSB_GATE_H0) QSB_CARRIER_KV(QSB_GATE_H0_FMA) QSB_CARRIER_KV(QSB_GATE_PAIR) \
QSB_CARRIER_KV(QSB_GLV_COEFF_BOUNDS) QSB_CARRIER_KV(QSB_GLV_FALLBACK_INLINE) \
QSB_CARRIER_KV(QSB_GLV_HIGH15) QSB_CARRIER_KV(QSB_GLV_RESIDUAL129) QSB_CARRIER_KV(QSB_GLV_RESIDUAL3) \
QSB_CARRIER_KV(QSB_GT_HEAL) QSB_CARRIER_KV(QSB_HOST_VERIFY) QSB_CARRIER_KV(QSB_HV_STATS) \
QSB_CARRIER_KV(QSB_ISO_FAST_X) QSB_CARRIER_KV(QSB_ISO_FUSED_ROOT_SCALE) \
QSB_CARRIER_KV(QSB_ISO_RELOAD_R) QSB_CARRIER_KV(QSB_ISO_ROOT_SCALE) \
QSB_CARRIER_KV(QSB_K2S_PARITY_NARROW) QSB_CARRIER_KV(QSB_K2S_PARITY_WINDOW) QSB_CARRIER_PW_HI_APPROX QSB_CARRIER_KV(QSB_K32) QSB_CARRIER_KV(QSB_K32_BGLUE) QSB_CARRIER_KV(QSB_SQR_X0_GLUE) \
QSB_CARRIER_KV(QSB_NEGFOLD_PARITY) QSB_CARRIER_KV(QSB_NEG_SHORT) QSB_CARRIER_KV(QSB_PAIR_SHARED) QSB_CARRIER_TREE_KARATSUBA QSB_CARRIER_INVERSE_PACK_CARRY QSB_CARRIER_GT_PREFIX_CACHE \
QSB_CARRIER_FIRST_DEVICE QSB_CARRIER_FIRST_LOCAL QSB_CARRIER_KV(QSB_PAIR_SHA_UNROLL_CONST) QSB_CARRIER_KV(QSB_PAIR_SHA_UNROLL_CONST_INNER) \
QSB_CARRIER_KV(QSB_PAIR_SHA_UNROLL_WINDOW) QSB_CARRIER_KV(QSB_GLV11) QSB_CARRIER_KV(QSB_GLV11_P18) QSB_CARRIER_KV(QSB_Q_P18) QSB_CARRIER_KV(QSB_Q_MIX) QSB_CARRIER_KV(QSB_Y_PAIR) QSB_CARRIER_KV(QSB_GLV_LEAN) QSB_CARRIER_KV(QSB_GLV_NO_KRED) QSB_CARRIER_KV(QSB_GLV_ROUND_CC) QSB_CARRIER_KV(QSB_GLV_HIGH15_HI) QSB_CARRIER_KV(QSB_GROUP_CAP_EXACT) QSB_CARRIER_KV(QSB_PREFIX_BLOCKS) \
QSB_CARRIER_KV(QSB_R_CBANK) QSB_CARRIER_KV(QSB_S3_HALF_WALK) QSB_CARRIER_KV(QSB_S3_SIGN_SHIFT) QSB_CARRIER_KV(QSB_S3_ODD_FOLD) QSB_CARRIER_KV(QSB_ROOT_MAX_BATCHES) QSB_CARRIER_KV(QSB_SHA_ALU_ADD) \
QSB_CARRIER_KV(QSB_SHA_FMA_ADD) QSB_CARRIER_KV(QSB_SHA_FMA_ROT) QSB_CARRIER_KV(QSB_SHA_UNROLL_CONST) \
QSB_CARRIER_KV(QSB_SHORT_CARRY) QSB_CARRIER_KV(QSB_SHORT_CARRY2) \
QSB_CARRIER_KV(QSB_SHORT_CARRY2_SENTINEL) QSB_CARRIER_KV(QSB_SHORT_CARRY3) \
QSB_CARRIER_KV(QSB_SHORT_CARRY4) QSB_CARRIER_KV(QSB_SHORT_CARRY6) QSB_CARRIER_KV(QSB_SLOT_PIPELINE) \
QSB_CARRIER_KV(QSB_SPEC_LAST_RESOLVE) QSB_CARRIER_KV(QSB_SPEC_PREPARE_PAIR) \
QSB_CARRIER_KV(QSB_STARTUP_TRIM) QSB_CARRIER_KV(QSB_TABLE_BASE_A) \
QSB_CARRIER_KV(QSB_TABLE_L2_WINDOW) QSB_CARRIER_KV(QSB_TRIM_DIRECT_PRODUCER) \
QSB_CARRIER_KV(QSB_Z2_SPEC_CUT) QSB_CARRIER_KV(ZLAB_DIRDIG) QSB_CARRIER_KV(ZLAB_DUAL_EPOCH_SHA) \
QSB_CARRIER_KV(ZLAB_HITPATH) QSB_CARRIER_KV(ZLAB_K2S3M) QSB_CARRIER_KV(ZLAB_LAUNCH_BLOCKS) \
QSB_CARRIER_KV(ZLAB_MODSQR) QSB_CARRIER_KV(ZLAB_PAIRSHA) QSB_CARRIER_KV(ZLAB_T14) \
QSB_CARRIER_KV(ZLAB_TREE) QSB_CARRIER_KV(ZLAB_TRIM) QSB_CARRIER_KV(QSB_FORCE_EXACT_HIT_CHECK) \
QSB_CARRIER_KV(QSB_SC_OPS) QSB_CARRIER_KV(QSB_SC_PP) QSB_CARRIER_KV(QSB_SC_ALUZ) QSB_CARRIER_KV(QSB_SC_PARK) QSB_CARRIER_KV(QSB_SC_LATE) \
QSB_CARRIER_KV(QSB_GATHER_LEA2) QSB_CARRIER_KV(QSB_Q_SPREAD) QSB_CARRIER_KV(QSB_GLV_EO) \
QSB_CARRIER_KV(QSB_GLV_RND) QSB_CARRIER_KV(QSB_GLV_ZDEC) QSB_CARRIER_KV(QSB_YP_FOLD_DROP) \
QSB_CARRIER_KV(QSB_GT_BATCH) QSB_CARRIER_KV(QSB_PSI_HOIST) QSB_CARRIER_KV(QSB_TREE_WAVE_TOP) \
QSB_CARRIER_KV(QSB_ROOT_LUT_SMEM) QSB_CARRIER_KV(QSB_PRE3_ROOT) QSB_CARRIER_KV(QSB_ROOT_WARP) \
QSB_CARRIER_KV(QSB_TAIL_STAGGER) QSB_CARRIER_KV(QSB_TAIL_PARK) QSB_CARRIER_KV(QSB_YNEG_FOLD) \
QSB_CARRIER_KV(QSB_SHA_SCHED_V4) QSB_CARRIER_KV(QSB_OUTER_LITK) QSB_CARRIER_KV(QSB_S3_NM_MASK) \
QSB_CARRIER_KV(QSB_SHA_CONST_IV) QSB_CARRIER_KV(QSB_SHA_CONST_PEEL) QSB_CARRIER_KV(QSB_GATE_W8_LEA) \
QSB_CARRIER_KV(QSB_SEED_K32_SUB) QSB_CARRIER_KV(QSB_GATHER_L1_POLICY) QSB_CARRIER_KV(QSB_GATHER_ONE_FORM) \
QSB_CARRIER_KV(QSB_HIT_NO_COMBO) QSB_CARRIER_KV(QSB_R_CBANK_TAILS) QSB_CARRIER_KV(QSB_TAIL_WEAVE) QSB_CARRIER_KV(QSB_DIVSTEP_LOOKAHEAD) \
QSB_CARRIER_KV(QSB_DECODE_CUT) QSB_CARRIER_KV(QSB_K32_SUBCUT) QSB_CARRIER_KV(QSB_K32_ADDCUT) \
QSB_CARRIER_KV(QSB_FX3_PRESUB) QSB_CARRIER_KV(QSB_FX3_PRESUB_EARLY) QSB_CARRIER_KV(QSB_S3_DOFF) QSB_CARRIER_KV(QSB_S3_UNIFORM_G) \
QSB_CARRIER_KV(QSB_OK_FOLD) QSB_CARRIER_KV(QSB_XNEG_BRANCH) QSB_CARRIER_KV(QSB_PW_QN) QSB_CARRIER_KV(QSB_TID_UNSIGNED) \
QSB_CARRIER_KV(QSB_S3_NM_SEED) QSB_CARRIER_KV(QSB_TREE_UNROLL) QSB_CARRIER_KV(QSB_PARK128) QSB_CARRIER_TREE_TOP_SHFL QSB_CARRIER_CTA_WINDOW_SPLIT QSB_CARRIER_ROOT_LUT32 QSB_CARRIER_ROOT_LUT40 QSB_CARRIER_FIRST_FLAT_FAST QSB_CARRIER_OUTER_PAIR QSB_CARRIER_ROOT_LUT_GLOBAL QSB_CARRIER_ROOT_PARK_B QSB_CARRIER_DEN_CROSS_PRE QSB_CARRIER_DEN_CROSS_IDLE
#ifdef QSB_CARRIER_BUILD
__device__ __constant__ char qsb_carrier_knobs[] = QSB_CARRIER_KNOBS;
#endif




#ifndef QSB_SP_REFILL_FIRST
#define QSB_SP_REFILL_FIRST 1
#endif

#if QSB_YOFF_S


static int qsb_gt_offset_pass(uint8_t *d_gt) {
{


enum { YS = GT_CHUNKS * 8 + 32 };
static uint8_t ys_pre[YS][64], ys_post[YS][64];
size_t ys_at[YS];
unsigned ys_seed = 0x9e3779b9u;
for (int t = 0; t < YS; t++) { int ch, i; gt_spot_sample(t, &ys_seed, &ch, &i); ys_at[t] = ((size_t)gt_offset(ch) + i) * 64; }
cudaError_t ye = cudaSuccess;
for (int t = 0; t < YS && ye == cudaSuccess; t++) ye = cudaMemcpy(ys_pre[t], d_gt + ys_at[t], 64, cudaMemcpyDeviceToHost);
const unsigned yb = (unsigned)((GT_TOTAL_ENTRIES + 255) / 256);
if (ye == cudaSuccess && !qsb_carrier_try(kernel_gt_offset_y, QK_YOFF, dim3(yb), dim3(256), (cudaStream_t)0, d_gt))
kernel_gt_offset_y<<<yb, 256>>>(d_gt);
if (ye == cudaSuccess) ye = cudaDeviceSynchronize();
if (ye == cudaSuccess) ye = cudaGetLastError();
for (int t = 0; t < YS && ye == cudaSuccess; t++) ye = cudaMemcpy(ys_post[t], d_gt + ys_at[t], 64, cudaMemcpyDeviceToHost);
if (ye != cudaSuccess) { fprintf(stderr, "GTable offset pass failed: %s\n", cudaGetErrorString(ye)); return 1; }
int ys_bad = 0;
for (int t = 0; t < YS; t++) {
uint64_t a[4], b[4], cy = 0x800001E8ULL;
memcpy(a, ys_pre[t] + 32, 32); memcpy(b, ys_post[t] + 32, 32);
int ok = memcmp(ys_pre[t], ys_post[t], 32) == 0;
for (int l = 0; l < 4; l++) { const uint64_t s = a[l] + cy; cy = s < cy; ok &= (s == b[l]); }
ys_bad += !ok;
}
if (ys_bad) { fprintf(stderr, "GTable offset pass check FAILED: %d of %d records are not raw + (K-1)/2\n", ys_bad, (int)YS); return 1; }
printf("  GTable offset ordinates: y += (K-1)/2 on %u records (%d records read back)\n", (unsigned)GT_TOTAL_ENTRIES, (int)YS);
}
return 0;
}
#endif

int main(int argc, char **argv) {
if (argc < 5) {
printf("Usage: %s <digest_rN.bin> <gpu_index> <sequence> <locktime> [total_gpus] [global_offset] [easy] [single_hash] [--tiles=PATH]\n", argv[0]);
printf("  total_gpus: total GPUs across ALL machines (default: local count)\n");
printf("  global_offset: this machine's GPU offset (default: 0)\n");
printf("  --tiles=PATH: balanced two-level partition for this GPU (overrides default mod-N partitioning)\n");
return 1;
}
int gpu_index = atoi(argv[2]);
uint32_t seq_val = (uint32_t)strtoul(argv[3], NULL, 0);
uint32_t lt_val = (uint32_t)strtoul(argv[4], NULL, 0);
int total_gpus_override = (argc >= 6) ? atoi(argv[5]) : 0;
int global_offset = (argc >= 7) ? atoi(argv[6]) : 0;
int easy = 0;
for (int i = 5; i < argc; i++) if (strcmp(argv[i], "easy") == 0) easy = 1;







int calibrate = 0;
for (int i = 5; i < argc; i++) if (strcmp(argv[i], "calibrate") == 0) calibrate = 1;
int single_hash = 0;
for (int i = 5; i < argc; i++) if (strcmp(argv[i], "single_hash") == 0) single_hash = 1;

const char *tile_path = NULL;
for (int i = 5; i < argc; i++) {
if (strncmp(argv[i], "--tiles=", 8) == 0) {
tile_path = argv[i] + 8;
}
}

#if QSB_LAZY_MODULES
setenv("CUDA_MODULE_LOADING", "LAZY", 0);
#endif
cudaSetDevice(gpu_index);
#if QSB_LAZY_MODULES
{
int rtv = 0, drv = 0;
cudaRuntimeGetVersion(&rtv); cudaDriverGetVersion(&drv);
const char *ml = getenv("CUDA_MODULE_LOADING");
const bool warn = rtv != 12080 || drv < 12080;
printf("  CUDA runtime %d.%d, driver %d.%d, module loading %s%s\n", rtv / 1000, rtv % 1000 / 10, drv / 1000, drv % 1000 / 10,
ml ? ml : "(unset)", !warn ? "" : drv < 12080 ? "; WARNING: driver older than the carrier's CUDA 12.8 toolkit, the native image will not load"
: "; WARNING: this build's CUDA runtime is not the carrier's 12.8 toolkit");
(void)cudaGetLastError();
}
#endif
#if QSB_L2_FETCH64



{
size_t l2g = 0;
cudaDeviceGetLimit(&l2g, cudaLimitMaxL2FetchGranularity);
const cudaError_t le = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64);
size_t l2n = l2g;
if (le == cudaSuccess) cudaDeviceGetLimit(&l2n, cudaLimitMaxL2FetchGranularity);
else (void)cudaGetLastError();
printf("  L2 fetch granularity: %zu -> %zu B\n", l2g, l2n);
}
#endif
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);
printf("QSB Digest Search [GPU %d]\n", gpu_index);
printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
qsb_carrier_init(prop, QSB_CARRIER_KNOBS);

digest_params_t dp;
if (load_digest_params(argv[1], &dp) < 0) return 1;
qsb_iso_params_t iso;
if(qsb_make_iso_params(&dp,&iso)<0)return 1;
printf("  Isomorphic recovery coordinates: xR'=%s1\n",iso.xneg?"-":"+");


int num_tiles = 0;
uint32_t *tile_first = NULL;
uint32_t *tile_lo = NULL;
uint32_t *tile_hi = NULL;
if (tile_path) {
FILE *tf = fopen(tile_path, "rb");
if (!tf) {
fprintf(stderr, "ERROR: cannot open tile file %s\n", tile_path);
return 1;
}
uint32_t n;
if (fread(&n, 4, 1, tf) != 1) { fprintf(stderr, "tile file truncated\n"); fclose(tf); return 1; }
num_tiles = (int)n;
tile_first = (uint32_t*)malloc(num_tiles * sizeof(uint32_t));
tile_lo = (uint32_t*)malloc(num_tiles * sizeof(uint32_t));
tile_hi = (uint32_t*)malloc(num_tiles * sizeof(uint32_t));
for (int i = 0; i < num_tiles; i++) {
uint32_t triple[3];
if (fread(triple, 4, 3, tf) != 3) {
fprintf(stderr, "tile file truncated at tile %d\n", i);
fclose(tf); return 1;
}
tile_first[i] = triple[0];
tile_lo[i] = triple[1];
tile_hi[i] = triple[2];
}
fclose(tf);
printf("  Loaded %d tiles from %s\n", num_tiles, tile_path);
}
















if (dp.tx_suffix_len >= 12) {
int seq_off = 0;
int lt_off = (int)dp.tx_suffix_len - 8;
dp.tx_suffix[seq_off + 0] = (seq_val ) & 0xFF;
dp.tx_suffix[seq_off + 1] = (seq_val >> 8) & 0xFF;
dp.tx_suffix[seq_off + 2] = (seq_val >> 16) & 0xFF;
dp.tx_suffix[seq_off + 3] = (seq_val >> 24) & 0xFF;
dp.tx_suffix[lt_off + 0] = (lt_val ) & 0xFF;
dp.tx_suffix[lt_off + 1] = (lt_val >> 8) & 0xFF;
dp.tx_suffix[lt_off + 2] = (lt_val >> 16) & 0xFF;
dp.tx_suffix[lt_off + 3] = (lt_val >> 24) & 0xFF;
printf("  Patched tx_suffix: seq=0x%08X (off=%d) lt=%u (off=%d) tx_suffix_len=%u\n",
seq_val, seq_off, lt_val, lt_off, dp.tx_suffix_len);
} else {
fprintf(stderr, "ERROR: tx_suffix_len=%u too short to patch seq+lt+sighash\n",
dp.tx_suffix_len);
return 2;
}




int n_pool = dp.n;
int t_sel = dp.t;









int window_start = 0, s_early = 0, t_win = t_sel;
uint64_t per_epoch = 0, n_epochs = 0;
int epoch_mode = 0;
int se_mode = 0;
{
int dev_count = 0;
cudaGetDeviceCount(&dev_count);
if (dev_count < 1) dev_count = 1;
int eff_total = (total_gpus_override > 0) ? total_gpus_override : dev_count;
if (tile_path == NULL && eff_total == 1 && !easy && !calibrate
&& n_pool == 150 && t_sel == 9
&& (int)dp.prefix_remainder_len == 42
&& (int)dp.tail_section_len == 218 && (int)dp.tx_suffix_len == 44
&& dp.total_preimage_len == 9906) {











se_mode = 1;
epoch_mode = 1;
window_start = QSB_SE_CUT; s_early = QSB_SE_EARLY; t_win = QSB_SE_TWIN;
per_epoch = QSB_SE_PER_EPOCH;
n_epochs = binom_u64(QSB_SE_CUT, QSB_SE_EARLY);
printf("  Short-epoch split: cut=%d, %d early omissions x %llu epochs, "
"%d window omissions x %d per epoch (%.3e candidates)\n",
window_start, s_early, (unsigned long long)n_epochs,
t_win, (int)per_epoch, (double)per_epoch * (double)n_epochs);
}
}
if (!se_mode && tile_path == NULL && total_gpus_override == 1 && t_sel >= 2 && n_pool > t_sel) {
const double SPACE_MIN = 4.0e11;
const double EPOCH_MIN = 1.0e6;
const int tail_suffix = (int)dp.tail_section_len + (int)dp.tx_suffix_len;
int best_blocks = 1 << 30;
double best_space = -1.0;
for (int s = 1; s < t_sel; s++) {
int tw = t_sel - s;
for (int cut = s; cut <= n_pool - tw; cut++) {
int K = n_pool - cut;
double per = binom_d(K, tw);
if (per < EPOCH_MIN) continue;
double space = per * binom_d(cut, s);
if (space < SPACE_MIN) continue;
int pre_bytes = (int)dp.prefix_remainder_len + (cut - s) * SIG_PUSH_SIZE;
int rem = pre_bytes % 64;
int varlen = rem + (K - tw) * SIG_PUSH_SIZE + tail_suffix;
int blocks = (varlen + 9 + 63) / 64;
if (blocks < best_blocks || (blocks == best_blocks && space > best_space)) {
best_blocks = blocks; best_space = space;
window_start = cut; s_early = s; t_win = tw;
}
}
}
if (best_space > 0.0) {
epoch_mode = 1;
per_epoch = binom_u64(n_pool - window_start, t_win);
n_epochs = binom_u64(window_start, s_early);
printf("  Epoch split: cut=%d, %d fixed early omissions x %llu epochs, "
"%d chosen from a %d-push window x %llu per epoch (%.3e candidates, "
"%d SHA blocks each)\n",
window_start, s_early, (unsigned long long)n_epochs,
t_win, n_pool - window_start, (unsigned long long)per_epoch,
best_space, best_blocks);
} else {
printf("  Epoch split: no split meets the space budget; using the full pool\n");
}
}

size_t launch_epochs = (size_t)QSB_SE_LAUNCH_BLOCKS * QSB_PAIR_MUL;
size_t group_capacity = qsb_group_capacity(window_start, s_early, launch_epochs);



uint8_t *epoch_prefix = NULL;
uint8_t epoch_rem[64];
uint8_t epoch_skip[MAX_T];
uint32_t epoch_mid[8];
int epoch_rem_len = 0;
memset(epoch_rem, 0, sizeof(epoch_rem));
memset(epoch_skip, 0, sizeof(epoch_skip));
if (epoch_mode) {
epoch_prefix = (uint8_t*)malloc(dp.prefix_remainder_len + (size_t)window_start * SIG_PUSH_SIZE + 64);
if (!epoch_prefix) { fprintf(stderr, "OOM: epoch prefix\n"); return 1; }
}







int fast_inc = 0;
int n_const_words = 0;
uint32_t *h_const_words = NULL;
if (epoch_mode) {
int n_inc = (n_pool - window_start) - t_win;
int pre_bytes = (int)dp.prefix_remainder_len + (window_start - s_early) * SIG_PUSH_SIZE;
int var_bytes = (pre_bytes % 64) + n_inc * SIG_PUSH_SIZE;
int stream = var_bytes + (int)dp.tail_section_len + (int)dp.tx_suffix_len;
int padded = ((stream + 9 + 63) / 64) * 64;
int const_bytes = padded - var_bytes;
int shape_ok;
if (se_mode) {




shape_ok = ((pre_bytes % 64) == 8 && (var_bytes % 4) == 0 && SIG_PUSH_SIZE == 10
&& t_win == QSB_SE_TWIN
&& n_inc == QSB_SE_N_INC && const_bytes == QSB_FAST_N_CONST * 4);
if (!shape_ok) {
fprintf(stderr, "ERROR: short-epoch shape mismatch "
"(kept=%d const_bytes=%d rem=%d); cannot run\n",
n_inc, const_bytes, pre_bytes % 64);
return 1;
}
} else {
shape_ok = ((pre_bytes % 64) == 0 && (var_bytes % 4) == 0 && SIG_PUSH_SIZE == 10
&& t_win <= 8
&& n_inc == QSB_FAST_N_INC && const_bytes == QSB_FAST_N_CONST * 4);
}
if (shape_ok) {
uint8_t *cb = (uint8_t*)calloc((size_t)const_bytes, 1);
if (!cb) { fprintf(stderr, "OOM: const words\n"); return 1; }
memcpy(cb, dp.tail_section, dp.tail_section_len);
memcpy(cb + dp.tail_section_len, dp.tx_suffix, dp.tx_suffix_len);
cb[dp.tail_section_len + dp.tx_suffix_len] = 0x80;
uint64_t bl = (uint64_t)dp.total_preimage_len * 8;
for (int i = 0; i < 8; i++) cb[const_bytes - 8 + i] = (uint8_t)(bl >> (56 - 8 * i));
n_const_words = const_bytes / 4;
h_const_words = (uint32_t*)malloc((size_t)const_bytes);
if (!h_const_words) { fprintf(stderr, "OOM: const words\n"); return 1; }
for (int i = 0; i < n_const_words; i++)
h_const_words[i] = ((uint32_t)cb[i*4]<<24)|((uint32_t)cb[i*4+1]<<16)
| ((uint32_t)cb[i*4+2]<<8)|(uint32_t)cb[i*4+3];
free(cb);
fast_inc = n_inc;
printf("  Register-resident assembly: %d kept pushes (%d message bytes) "
"+ %d constant words, %d SHA-256 blocks per candidate\n",
n_inc, var_bytes, n_const_words, padded / 64);
} else if (!se_mode) {
printf("  Register-resident assembly: shape mismatch "
"(kept=%d const_bytes=%d rem=%d); using the generic path\n",
n_inc, const_bytes, pre_bytes % 64);
}
}

if(fast_inc==QSB_FAST_N_INC||fast_inc==QSB_SE_N_INC) {
if (qsb_prepare_push_words(dp.dummy_sigs,n_pool)) {



if (se_mode) { fprintf(stderr,"ERROR: push-word prep failed\n"); return 1; }
fast_inc = 0;
} else if (qsb_prepare_constant_schedule(h_const_words,n_const_words)) {
return 1;
}
}
size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;
uint8_t *d_gt;
#if QSB_S3
{
cudaError_t gt_alloc=cudaMalloc(&d_gt,gt_sz);
if(gt_alloc!=cudaSuccess) {
fprintf(stderr,"GLV12 table allocation failed: %s\n",cudaGetErrorString(gt_alloc));
return 1;
}
}
#else
cudaMalloc(&d_gt,gt_sz);
#endif
{





struct timespec ta, tb; clock_gettime(CLOCK_MONOTONIC, &ta);
#if QSB_GT_PHASE_AUDIT
struct timespec gt_ladders_end,gt_kernel_end,gt_heal_end;
#endif
size_t lb = (size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t);
size_t hb = (size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t);
uint64_t *hL=(uint64_t*)malloc(lb), *hH=(uint64_t*)malloc(hb);
if(!hL||!hH){ fprintf(stderr,"OOM: gtable ladders\n"); return 1; }
#if QSB_S3
size_t h2b = (size_t)GT_CHUNKS*GT_H2*8*sizeof(uint64_t);
uint64_t *hH2=(uint64_t*)malloc(h2b);
if(!hH2){ fprintf(stderr,"OOM: gtable ladders\n"); return 1; }
#if QSB_S1_LADDER_SELFTEST
{
uint64_t *sL=(uint64_t*)malloc(lb), *sH=(uint64_t*)malloc(hb), *sH2=(uint64_t*)malloc(h2b);
struct timespec s0, s1, s2; clock_gettime(CLOCK_MONOTONIC, &s0);
gt_build_ladders(sL,sH,sH2,dp.neg_r_inv,iso.alpha,iso.beta);
clock_gettime(CLOCK_MONOTONIC, &s1);
gt_build_ladders_mt(hL,hH,hH2,dp.neg_r_inv,iso.alpha,iso.beta);
clock_gettime(CLOCK_MONOTONIC, &s2);
const int same = !memcmp(sL,hL,lb) && !memcmp(sH,hH,hb) && !memcmp(sH2,hH2,h2b);
printf("  S1 ladder selftest: sequential %.3f s, threaded %.3f s, %zu + %zu + %zu bytes %s\n",
(s1.tv_sec-s0.tv_sec)+(s1.tv_nsec-s0.tv_nsec)/1e9, (s2.tv_sec-s1.tv_sec)+(s2.tv_nsec-s1.tv_nsec)/1e9,
lb, hb, h2b, same ? "IDENTICAL" : "DIFFER");
free(sL); free(sH); free(sH2);
if (!same) return 3;
}
#elif QSB_STARTUP_THREADS
gt_build_ladders_mt(hL,hH,hH2,dp.neg_r_inv,iso.alpha,iso.beta);
#else
gt_build_ladders(hL,hH,hH2,dp.neg_r_inv,iso.alpha,iso.beta);
#endif
uint64_t *dH2=NULL; cudaMalloc(&dH2,h2b);
cudaMemcpy(dH2,hH2,h2b,cudaMemcpyHostToDevice);
free(hH2);
#else
gt_build_ladders(hL,hH,dp.neg_r_inv,iso.alpha,iso.beta);
#endif
uint64_t *dL=NULL,*dH=NULL; cudaMalloc(&dL,lb); cudaMalloc(&dH,hb);
cudaMemcpy(dL,hL,lb,cudaMemcpyHostToDevice);
cudaMemcpy(dH,hH,hb,cudaMemcpyHostToDevice);
free(hL); free(hH);
int gt_total = GT_TOTAL_ENTRIES;
#if QSB_GT_PHASE_AUDIT
clock_gettime(CLOCK_MONOTONIC,&gt_ladders_end);
#endif
#if QSB_S3
#if QSB_GT_BATCH

const unsigned gt_blocks = (unsigned)((GT_BATCH_THREADS + 255) / 256);
if (!qsb_carrier_try(kernel_build_gtable, QK_GT, dim3(gt_blocks), dim3(256), (cudaStream_t)0,
dL, dH, dH2, d_gt))
kernel_build_gtable<<<gt_blocks,256>>>(dL,dH,dH2,d_gt);
#else
if (!qsb_carrier_try(kernel_build_gtable, QK_GT, dim3((gt_total+255)/256), dim3(256), (cudaStream_t)0,
dL, dH, dH2, d_gt))
kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,dH2,d_gt);
#endif
cudaError_t gerr = cudaDeviceSynchronize();
if(gerr==cudaSuccess) gerr=cudaGetLastError();
cudaFree(dL); cudaFree(dH); cudaFree(dH2);
#else
if (!qsb_carrier_try(kernel_build_gtable, QK_GT, dim3((gt_total+255)/256), dim3(256), (cudaStream_t)0,
dL, dH, d_gt))
kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,d_gt);
cudaDeviceSynchronize();
cudaError_t gerr = cudaGetLastError();
cudaFree(dL); cudaFree(dH);
#endif
uint8_t *chk_table=NULL;
#if QSB_GT_PHASE_AUDIT
clock_gettime(CLOCK_MONOTONIC,&gt_kernel_end);
#endif
int gt_ok = (gerr==cudaSuccess);
#if QSB_S3 && QSB_GT_HEAL
if(gt_ok){
unsigned n_flagged=0, n_rewritten=0;
gt_ok = gt_heal(d_gt,dp.neg_r_inv,iso.alpha,iso.beta,&n_flagged,&n_rewritten)==0;
printf("  GTable heal: %u off-curve flags, %u records rewritten from OpenSSL%s\n",
n_flagged, n_rewritten, gt_ok ? "" : " (heal failed)");
}
#endif
#if QSB_STARTUP_TRIM
if (gt_ok) {
#if QSB_GT_PHASE_AUDIT
clock_gettime(CLOCK_MONOTONIC,&gt_heal_end);
#endif
#if QSB_S3
const int samples = GT_CHUNKS*8+192;
#else
const int samples = GT_CHUNKS*4+192;
#endif
uint8_t *h_samp=NULL;
const int pinned=(cudaHostAlloc((void**)&h_samp,(size_t)samples*64,cudaHostAllocDefault)==cudaSuccess);
if (!pinned) { h_samp=(uint8_t*)malloc((size_t)samples*64); if(!h_samp){fprintf(stderr,"OOM: gtable check\n");return 1;} }
unsigned seed=0x9e3779b9u;
cudaError_t ce=cudaSuccess;
for (int t=0;t<samples && ce==cudaSuccess;t++) {
int ch,i; gt_spot_sample(t,&seed,&ch,&i);
const uint8_t *srcp=d_gt+((size_t)gt_offset(ch)+i)*64;
ce=pinned ? cudaMemcpyAsync(h_samp+(size_t)t*64,srcp,64,cudaMemcpyDeviceToHost,0)
: cudaMemcpy(h_samp+(size_t)t*64,srcp,64,cudaMemcpyDeviceToHost);
}
if (ce==cudaSuccess) ce=cudaDeviceSynchronize();
#if QSB_S1_LADDER_SELFTEST && QSB_S3
{
struct timespec s0, s1, s2; clock_gettime(CLOCK_MONOTONIC, &s0);
const int a = (ce==cudaSuccess) && gt_spot_check(NULL,samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
clock_gettime(CLOCK_MONOTONIC, &s1);
const int b = (ce==cudaSuccess) && gt_spot_check_mt(samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
clock_gettime(CLOCK_MONOTONIC, &s2);
h_samp[(size_t)(samples/2)*64+7] ^= 0x10;
const int ca = gt_spot_check(NULL,samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
const int cb = gt_spot_check_mt(samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
printf("  S1 spot-check selftest: %d samples, sequential %d (%.3f s), threaded %d (%.3f s); one corrupted sample: sequential %d, threaded %d\n",
samples, a, (s1.tv_sec-s0.tv_sec)+(s1.tv_nsec-s0.tv_nsec)/1e9, b,
(s2.tv_sec-s1.tv_sec)+(s2.tv_nsec-s1.tv_nsec)/1e9, ca, cb);
fflush(stdout);
_exit((a==1 && b==1 && ca==0 && cb==0) ? 0 : 4);
}
#elif QSB_STARTUP_THREADS && QSB_S3
gt_ok=(ce==cudaSuccess) && gt_spot_check_mt(samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
#else
gt_ok=(ce==cudaSuccess) && gt_spot_check(NULL,samples,dp.neg_r_inv,iso.alpha,iso.beta,h_samp);
#endif
if (pinned) cudaFreeHost(h_samp); else free(h_samp);
}
#else
chk_table=(uint8_t*)malloc(gt_sz);
if(!chk_table){ fprintf(stderr,"OOM: gtable check\n"); return 1; }
if(gt_ok){
cudaMemcpy(chk_table,d_gt,gt_sz,cudaMemcpyDeviceToHost);
gt_ok = gt_spot_check(chk_table,GT_CHUNKS*4+192,dp.neg_r_inv,iso.alpha,iso.beta,NULL);
}
#endif
clock_gettime(CLOCK_MONOTONIC, &tb);
double gt_secs=(tb.tv_sec-ta.tv_sec)+(tb.tv_nsec-ta.tv_nsec)/1e9;
#if QSB_GT_PHASE_AUDIT
if(!gt_ok){fprintf(stderr,"GT_PHASE_AUDIT FAIL: existing build/heal/spot checks\n");return 3;}
const auto phase_s=[](const timespec &a,const timespec &b){return (b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;};
printf("GT_PHASE_AUDIT ladders_s=%.9f kernel_s=%.9f heal_s=%.9f spot_s=%.9f total_s=%.9f\n",
phase_s(ta,gt_ladders_end),phase_s(gt_ladders_end,gt_kernel_end),phase_s(gt_kernel_end,gt_heal_end),phase_s(gt_heal_end,tb),gt_secs);
SHA256_CTX gt_hash;SHA256_Init(&gt_hash);
const size_t gt_piece=16u*1024u*1024u;uint8_t *gt_buf=(uint8_t*)malloc(gt_piece);
if(!gt_buf)return 4;
for(size_t at=0;at<gt_sz;at+=gt_piece){const size_t n=gt_sz-at<gt_piece?gt_sz-at:gt_piece;
if(cudaMemcpy(gt_buf,d_gt+at,n,cudaMemcpyDeviceToHost)!=cudaSuccess){free(gt_buf);return 5;}
SHA256_Update(&gt_hash,gt_buf,n);}
unsigned char gt_digest[32];SHA256_Final(gt_digest,&gt_hash);free(gt_buf);
printf("GT_PHASE_AUDIT table_bytes=%zu sha256=",gt_sz);for(int i=0;i<32;i++)printf("%02x",gt_digest[i]);
printf(" build_heal_spot=PASS\n");fflush(stdout);
FILE *gt_receipt=fopen("results/gt_phase_audit.txt","w");
if(!gt_receipt){fprintf(stderr,"GT_PHASE_AUDIT FAIL: receipt open\n");return 6;}
fprintf(gt_receipt,"ladders_s=%.9f kernel_s=%.9f heal_s=%.9f spot_s=%.9f total_s=%.9f table_bytes=%zu prefix_cache=%d sha256=",
phase_s(ta,gt_ladders_end),phase_s(gt_ladders_end,gt_kernel_end),phase_s(gt_kernel_end,gt_heal_end),phase_s(gt_heal_end,tb),gt_secs,gt_sz,QSB_GT_PREFIX_CACHE);
for(int i=0;i<32;i++)fprintf(gt_receipt,"%02x",gt_digest[i]);
fprintf(gt_receipt," build_heal_spot=PASS\n");
if(fclose(gt_receipt)){fprintf(stderr,"GT_PHASE_AUDIT FAIL: receipt close\n");return 7;}
#endif
if(gt_ok){
printf("  GTable built on GPU in %.2fs (%d points, %.0f MiB total, spot check passed)\n",
gt_secs, gt_total, (double)gt_sz/(1024*1024));
} else {
printf("  GTable GPU build rejected (%s); using the host builder\n",
gerr!=cudaSuccess ? cudaGetErrorString(gerr) : "spot check failed");
#if QSB_STARTUP_TRIM
chk_table=(uint8_t*)malloc(gt_sz);
if(!chk_table){fprintf(stderr,"OOM: gtable fallback\n");return 1;}
#endif
compute_gtable(chk_table,dp.neg_r_inv,iso.alpha,iso.beta);
cudaMemcpy(d_gt,chk_table,gt_sz,cudaMemcpyHostToDevice);
}
fflush(stdout);
free(chk_table);
}

#if QSB_YOFF_S
if (qsb_gt_offset_pass(d_gt)) return 1;
#endif


uint32_t *d_mid; cudaMalloc(&d_mid,32);
cudaMemcpy(d_mid, dp.midstate, 32, cudaMemcpyHostToDevice);


uint8_t *d_prem = NULL;
if (epoch_mode || dp.prefix_remainder_len > 0) {
cudaMalloc(&d_prem, 64);
if (dp.prefix_remainder_len > 0)
cudaMemcpy(d_prem, dp.prefix_remainder, dp.prefix_remainder_len, cudaMemcpyHostToDevice);
}
uint8_t *d_early = NULL;
cudaMalloc(&d_early, MAX_T);
cudaMemset(d_early, 0, MAX_T);
uint32_t *d_const_words = NULL;
cudaMalloc(&d_const_words, (n_const_words ? n_const_words : 1) * sizeof(uint32_t));
if (n_const_words)
cudaMemcpy(d_const_words, h_const_words, n_const_words * sizeof(uint32_t),
cudaMemcpyHostToDevice);
uint8_t *d_dsigs; cudaMalloc(&d_dsigs, n_pool*SIG_PUSH_SIZE);
cudaMemcpy(d_dsigs, dp.dummy_sigs, n_pool*SIG_PUSH_SIZE, cudaMemcpyHostToDevice);
uint8_t *d_tail; cudaMalloc(&d_tail, dp.tail_section_len);
cudaMemcpy(d_tail, dp.tail_section, dp.tail_section_len, cudaMemcpyHostToDevice);
uint8_t *d_suf; cudaMalloc(&d_suf, dp.tx_suffix_len);
cudaMemcpy(d_suf, dp.tx_suffix, dp.tx_suffix_len, cudaMemcpyHostToDevice);






epoch_desc_t *d_epochs = NULL;
#if QSB_EPOCH_GROUPS
qsb_group_t *d_groups = NULL;
#if QSB_EPOCH_GROUPS && QSB_EPOCH_FAST
uint32_t *d_epoch_group = NULL;
#endif
#endif
uint32_t *d_first = NULL;
#if QSB_SLOT_PIPELINE

epoch_desc_t *d_epochs_s[2] = {NULL, NULL};
uint32_t *d_first_s[2] = {NULL, NULL};
#if QSB_EPOCH_GROUPS
qsb_group_t *d_groups_s[2] = {NULL, NULL};
#if QSB_EPOCH_FAST
uint32_t *d_epoch_group_s[2] = {NULL, NULL};
#endif
#endif
#endif
if (se_mode) {
#if QSB_LAUNCH_BUDGET



if (QSB_SLOT_PIPELINE && ZLAB_HITPATH && QSB_EPOCH_GROUPS && QSB_EPOCH_FAST &&
QSB_GROUP_CAP_TIGHT && !QSB_FIRST_PRODUCER_SCRATCH && !QSB_FIRST_SLOT_INTERLEAVE &&
QSB_SE_WINDOWS == 128 && QSB_FIRST_SLOTS == 16 &&
window_start == 137 && s_early == 6 &&
(launch_epochs == 1048576 || launch_epochs == 2097152)) {
size_t free_bytes = 0, total_bytes = 0;
const cudaError_t budget_rc = cudaMemGetInfo(&free_bytes, &total_bytes);
if (budget_rc != cudaSuccess) {
fprintf(stderr,"ERROR: launch memory query: %s\n",cudaGetErrorString(budget_rc)); return 1;
}
const size_t reserve = (size_t)QSB_LAUNCH_RESERVE_MIB * 1048576;
const size_t selected = qsb_select_launch_epochs(launch_epochs, free_bytes, reserve,
(size_t)QSB_LAUNCH_TEST_MAX_EPOCHS, sizeof(epoch_desc_t), sizeof(qsb_group_t),
(size_t)QSB_FIRST_SLOTS * 32,
[&](size_t e) { return qsb_group_capacity(window_start, s_early, e); });
if (!selected) {
fprintf(stderr,"OOM: launch slots cannot fit minimum with %zu-byte reserve (free=%zu)\n",reserve,free_bytes);
return 1;
}
const size_t old_need = qsb_launch_buffer_bytes(launch_epochs, group_capacity,
sizeof(epoch_desc_t), sizeof(qsb_group_t), (size_t)QSB_FIRST_SLOTS * 32);
launch_epochs = selected;
group_capacity = qsb_group_capacity(window_start, s_early, launch_epochs);
const size_t new_need = qsb_launch_buffer_bytes(launch_epochs, group_capacity,
sizeof(epoch_desc_t), sizeof(qsb_group_t), (size_t)QSB_FIRST_SLOTS * 32);
fprintf(stderr,"LAUNCH_BUDGET epochs=%zu slot_bytes=%zu saved=%zu free=%zu reserve=%zu total=%zu\n",
launch_epochs,new_need,old_need-new_need,free_bytes,reserve,total_bytes);
}
#endif
uint8_t h_win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
int cnt = 0;
for (int a = 0; a < 13; a++)
for (int b = a + 1; b < 13; b++)
for (int c = b + 1; c < 13; c++) {
#if QSB_SE_WINDOWS == 256


if(a>=1 && c<=7 && !(a==1 && b==2))continue;
#elif QSB_SE_WINDOWS == 128






if(!((a>=6) || (a<=5 && b>=7) || (a==0 && b==1 && c>=8 && c<=10)))continue;
#else
#error "QSB_SE_WINDOWS must be 128 or 256"
#endif
h_win3[cnt][0] = (uint8_t)(QSB_SE_CUT + a);
h_win3[cnt][1] = (uint8_t)(QSB_SE_CUT + b);
h_win3[cnt][2] = (uint8_t)(QSB_SE_CUT + c);
cnt++;
}
if(cnt!=QSB_SE_PER_EPOCH)return 1;


for(int i=1;i<QSB_SE_PER_EPOCH;i++){
uint8_t w[3];memcpy(w,h_win3[i],3);
uint32_t second=qsb_window_second_key(w),first=qsb_window_first_key(w);int j=i;
while(j>0 && (qsb_window_second_key(h_win3[j-1])>second ||
(qsb_window_second_key(h_win3[j-1])==second && qsb_window_first_key(h_win3[j-1])>first))){
memcpy(h_win3[j],h_win3[j-1],3);j--;
}
memcpy(h_win3[j],w,3);
}
QSB_TO_SYMBOL(WIN3, h_win3, sizeof(h_win3));
#if QSB_HOST_VERIFY
memcpy(g_hv_win3, h_win3, sizeof(h_win3));
#endif
if (qsb_prepare_window_schedule(dp.dummy_sigs, h_win3, h_const_words)) return 1;
#ifdef QSB_HP_ON
qhp::start(&dp, window_start, s_early, qsb_first_class_count, n_epochs,
(uint64_t)launch_epochs);
#endif
cudaMalloc(&d_epochs, launch_epochs * sizeof(epoch_desc_t));
if (!d_epochs) { fprintf(stderr, "OOM: epoch descriptors\n"); return 1; }
#if QSB_FIRST_PRODUCER_SCRATCH && QSB_SLOT_PIPELINE && ZLAB_HITPATH && QSB_EPOCH_GROUPS && QSB_EPOCH_FAST
const size_t scratch_epochs = launch_epochs;
const size_t first_row_bytes = (size_t)QSB_FIRST_SLOTS * 8 * sizeof(uint32_t);
if (scratch_epochs > SIZE_MAX / first_row_bytes ||
scratch_epochs > SIZE_MAX / sizeof(uint32_t) ||
group_capacity > (SIZE_MAX - 255) / sizeof(qsb_group_t)) {
fprintf(stderr, "ERROR: producer scratch size overflow\n"); return 1;
}
const size_t scratch_group_bytes = group_capacity * sizeof(qsb_group_t);
const size_t scratch_map_offset = (scratch_group_bytes + 255) & ~(size_t)255;
const size_t scratch_map_bytes = scratch_epochs * sizeof(uint32_t);
const size_t scratch_first_bytes = scratch_epochs * first_row_bytes;
const bool producer_overlay = scratch_map_offset <= scratch_first_bytes &&
scratch_map_bytes <= scratch_first_bytes - scratch_map_offset;
if (producer_overlay) {
cudaError_t se = cudaMalloc(&d_first, scratch_first_bytes);
if (se != cudaSuccess) {
fprintf(stderr, "OOM: first producer scratch: %s\n", cudaGetErrorString(se)); return 1;
}
d_groups = reinterpret_cast<qsb_group_t *>(d_first);
d_epoch_group = reinterpret_cast<uint32_t *>(
reinterpret_cast<uint8_t *>(d_first) + scratch_map_offset);
fprintf(stderr, "FIRST_PRODUCER_SCRATCH slots=2 groups=%zu map=%zu offset=%zu first=%zu saved=%zu bytes\n",
scratch_group_bytes, scratch_map_bytes, scratch_map_offset,
scratch_first_bytes, 2 * (scratch_group_bytes + scratch_map_bytes));
}
#endif
#if QSB_EPOCH_GROUPS


#if QSB_FIRST_PRODUCER_SCRATCH && QSB_SLOT_PIPELINE && ZLAB_HITPATH && QSB_EPOCH_FAST
if (!producer_overlay)
#endif
cudaMalloc(&d_groups, group_capacity * sizeof(qsb_group_t));
#if QSB_EPOCH_GROUPS && QSB_EPOCH_FAST
#if QSB_FIRST_PRODUCER_SCRATCH && QSB_SLOT_PIPELINE && ZLAB_HITPATH
if (!producer_overlay)
#endif
cudaMalloc(&d_epoch_group, launch_epochs * sizeof(uint32_t));
if(!d_epoch_group){fprintf(stderr,"OOM: epoch-group map\n");return 1;}
#endif
if (!d_groups) { fprintf(stderr, "OOM: epoch groups\n"); return 1; }
#endif
cudaError_t first_error=cudaSuccess;
#if QSB_FIRST_PRODUCER_SCRATCH && QSB_SLOT_PIPELINE && ZLAB_HITPATH && QSB_EPOCH_GROUPS && QSB_EPOCH_FAST
if (!producer_overlay)
#endif
first_error=cudaMalloc(&d_first,launch_epochs*QSB_FIRST_SLOTS*8*sizeof(uint32_t));
if(first_error!=cudaSuccess){fprintf(stderr,"OOM: first states: %s\n",cudaGetErrorString(first_error));return 1;}
#if QSB_SLOT_PIPELINE
d_epochs_s[0]=d_epochs; d_first_s[0]=d_first;
{
cudaError_t se=cudaMalloc(&d_epochs_s[1],launch_epochs*sizeof(epoch_desc_t));
#if QSB_FIRST_SLOT_INTERLEAVE && !QSB_FIRST_PRODUCER_SCRATCH




const bool first_interleave = QSB_SE_WINDOWS == 128 &&
QSB_FIRST_SLOTS == 16 && qsb_first_class_count == 8;
if (se==cudaSuccess && first_interleave) {
d_first_s[1]=d_first + (QSB_FIRST_SLOTS/2)*8;
fprintf(stderr,"FIRST_SLOT_INTERLEAVE pitch=%zu width=%zu offset=%zu saved=%zu bytes\n",
(size_t)QSB_FIRST_SLOTS*32, (size_t)qsb_first_class_count*32,
(size_t)(QSB_FIRST_SLOTS/2)*32,
launch_epochs*QSB_FIRST_SLOTS*32);
}
if (!first_interleave)
#endif
if(se==cudaSuccess) se=cudaMalloc(&d_first_s[1],launch_epochs*QSB_FIRST_SLOTS*8*sizeof(uint32_t));
#if QSB_EPOCH_GROUPS
d_groups_s[0]=d_groups;
#if QSB_FIRST_PRODUCER_SCRATCH && ZLAB_HITPATH && QSB_EPOCH_FAST
if (se==cudaSuccess && producer_overlay) {
d_groups_s[1]=reinterpret_cast<qsb_group_t *>(d_first_s[1]);
d_epoch_group_s[1]=reinterpret_cast<uint32_t *>(
reinterpret_cast<uint8_t *>(d_first_s[1]) + scratch_map_offset);
}
if (!producer_overlay)
#endif
if(se==cudaSuccess) se=cudaMalloc(&d_groups_s[1],group_capacity*sizeof(qsb_group_t));
#if QSB_EPOCH_FAST
d_epoch_group_s[0]=d_epoch_group;
#if QSB_FIRST_PRODUCER_SCRATCH && ZLAB_HITPATH
if (!producer_overlay)
#endif
if(se==cudaSuccess) se=cudaMalloc(&d_epoch_group_s[1],launch_epochs*sizeof(uint32_t));
#endif
#endif
if(se!=cudaSuccess){fprintf(stderr,"OOM: pipeline slot 1: %s\n",cudaGetErrorString(se));return 1;}
}
#endif
}

uint64_t *d_nri,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
cudaMalloc(&d_nri,32);cudaMalloc(&d_u2rx,32);cudaMalloc(&d_u2ry,32);
cudaMalloc(&d_neg2u2rx,32);cudaMalloc(&d_neg2u2ry,32);
cudaMemcpy(d_nri,dp.neg_r_inv,32,cudaMemcpyHostToDevice);
cudaMemcpy(d_u2rx,dp.u2r_x,32,cudaMemcpyHostToDevice);
cudaMemcpy(d_u2ry,dp.u2r_y,32,cudaMemcpyHostToDevice);
uint64_t h_u2r[8];
memcpy(h_u2r,dp.u2r_x,32);memcpy(h_u2r+4,dp.u2r_y,32);
if(QSB_TO_SYMBOL(QSB_U2R,h_u2r,sizeof(h_u2r))!=cudaSuccess){
fprintf(stderr,"ERROR: QSB_U2R upload failed\n");return 1;
}
if(QSB_TO_SYMBOL(QSB_U2R_ISO,iso.u2r_iso,sizeof(iso.u2r_iso))!=cudaSuccess ||
QSB_TO_SYMBOL(QSB_ISO_INVU,iso.invu,sizeof(iso.invu))!=cudaSuccess ||
QSB_TO_SYMBOL(QSB_ISO_XNEG,&iso.xneg,sizeof(iso.xneg))!=cudaSuccess){
fprintf(stderr,"ERROR: isomorphic constants upload failed\n");return 1;
}

{
static const uint8_t p_be[32]={
0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F};
BN_CTX *ctx=BN_CTX_new();
BIGNUM *bp=BN_new(),*bx=BN_new(),*by=BN_new(),*bc=BN_new(),*b3=BN_new();
BN_bin2bn(p_be,32,bp);
BN_lebin2bn(dp.u2r_x,32,bx);
BN_lebin2bn(dp.u2r_y,32,by);
BN_mod_add(by,by,by,bp,ctx);
if(BN_mod_inverse(by,by,bp,ctx)==NULL){fprintf(stderr,"ERROR: QSB_U2R_C inverse failed\n");return 1;}
BN_mod_sqr(bc,bx,bp,ctx);
BN_set_word(b3,3);
BN_mod_mul(bc,bc,b3,bp,ctx);
BN_mod_mul(bc,bc,by,bp,ctx);
uint64_t h_c[4 + 4*(QSB_K2S_CENTER_SQR != 0) + (QSB_K2S_CENTER_FUSED == 2)];
if(BN_bn2lebinpad(bc,(uint8_t*)h_c,32)!=32){fprintf(stderr,"ERROR: QSB_U2R_C encode failed\n");return 1;}
#if QSB_K2S_CENTER_SQR
if(!BN_mod_sqr(bc,bc,bp,ctx)){
fprintf(stderr,"ERROR: QSB_U2R_C squared failed\n");return 1;
}
#if QSB_K2S_CENTER_FUSED == 2

if(!BN_copy(b3,bp) || !BN_mul_word(b3,3) || !BN_sub(bc,b3,bc) ||
BN_bn2lebinpad(bc,(uint8_t*)(h_c+4),40)!=40){
fprintf(stderr,"ERROR: QSB_U2R_C prebias encode failed\n");return 1;
}
#else
if(BN_bn2lebinpad(bc,(uint8_t*)(h_c+4),32)!=32){
fprintf(stderr,"ERROR: QSB_U2R_C squared encode failed\n");return 1;
}
#endif
#endif
if(QSB_TO_SYMBOL(QSB_U2R_C,h_c,sizeof(h_c))!=cudaSuccess){
fprintf(stderr,"ERROR: QSB_U2R_C upload failed\n");return 1;
}
BN_free(bp);BN_free(bx);BN_free(by);BN_free(bc);BN_free(b3);BN_CTX_free(ctx);
}


{
EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *ctx=BN_CTX_new();
BIGNUM *bx=BN_new(),*by=BN_new();
uint8_t be[32];
for(int i=0;i<32;i++) be[i]=dp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
for(int i=0;i<32;i++) be[i]=dp.u2r_y[31-i]; BN_bin2bn(be,32,by);
EC_POINT *pt=EC_POINT_new(grp);
EC_POINT_set_affine_coordinates_GFp(grp,pt,bx,by,ctx);
EC_POINT *dbl=EC_POINT_new(grp);
EC_POINT_dbl(grp,dbl,pt,ctx);
EC_POINT_invert(grp,dbl,ctx);
BIGNUM *dx=BN_new(),*dy=BN_new();
EC_POINT_get_affine_coordinates_GFp(grp,dbl,dx,dy,ctx);
uint8_t dxb[32],dyb[32]; memset(dxb,0,32);memset(dyb,0,32);
BN_bn2bin(dx,dxb+(32-BN_num_bytes(dx)));
BN_bn2bin(dy,dyb+(32-BN_num_bytes(dy)));
uint64_t n2x[4],n2y[4];
for(int i=0;i<4;i++){n2x[i]=0;n2y[i]=0;
for(int b=0;b<8;b++){n2x[i]|=(uint64_t)dxb[31-i*8-b]<<(b*8);
n2y[i]|=(uint64_t)dyb[31-i*8-b]<<(b*8);}}
cudaMemcpy(d_neg2u2rx,n2x,32,cudaMemcpyHostToDevice);
cudaMemcpy(d_neg2u2ry,n2y,32,cudaMemcpyHostToDevice);
BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
EC_POINT_free(pt);EC_POINT_free(dbl);
EC_GROUP_free(grp);BN_CTX_free(ctx);
}

#if !QSB_STARTUP_TRIM
cudaDeviceSetLimit(cudaLimitStackSize, 32768);
#endif
uint32_t *d_hit_cnt, *d_hit_idx;
uint8_t *d_hit_combos, *d_hit_sighash;
uint8_t *d_hit_keynonce, *d_hit_pubhash, *d_hit_qx, *d_hit_qy;
cudaMalloc(&d_hit_cnt,4);cudaMalloc(&d_hit_idx,1024*4);
cudaMalloc(&d_hit_combos, 1024 * MAX_T);
cudaMalloc(&d_hit_sighash, 1024 * 32);
cudaMalloc(&d_hit_keynonce, 1024 * 33);
cudaMalloc(&d_hit_pubhash, 1024 * 32);
cudaMalloc(&d_hit_qx, 1024 * 32);
cudaMalloc(&d_hit_qy, 1024 * 32);

int BATCH = 8388608;
int BLKSZ = 256;


int num_gpus = 0;
cudaGetDeviceCount(&num_gpus);
if (num_gpus < 1) num_gpus = 1;


int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
int effective_id = global_offset + gpu_index;

printf("  Mode: %s, GPU %d (global %d of %d)\n", easy?"EASY":"REAL", gpu_index, effective_id, effective_total);
printf("  Batch: %d combos per kernel launch\n", BATCH);

uint8_t *h_combos = nullptr, *d_combos = nullptr;
#if !(QSB_TRIM_COMBO_ALLOC && ZLAB_TRIM)
h_combos = (uint8_t*)malloc(BATCH * t_sel);
cudaMalloc(&d_combos, BATCH * t_sel);
#endif


{
static uint64_t h_binom[151][10];
for (int n = 0; n <= 150; n++) {
for (int k = 0; k <= 9; k++) {
if (k > n) h_binom[n][k] = 0;
else if (k == 0 || k == n) h_binom[n][k] = 1;
else {
int kk = k; if (kk > n - kk) kk = n - kk;
__uint128_t r = 1;
for (int i = 0; i < kk; i++) {
r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
if (r > (uint64_t)0x7FFFFFFFFFFFFFFFULL) { r = (uint64_t)0x7FFFFFFFFFFFFFFFULL; break; }
}
h_binom[n][k] = (uint64_t)r;
}
}
}
QSB_TO_SYMBOL(BINOM_C, h_binom, sizeof(h_binom));
}








{
int debug_idx = -1;
for (int i = 5; i < argc; i++) {
if (strcmp(argv[i], "debug") == 0) { debug_idx = i; break; }
}


}

struct timespec t0, t1, t_last_report;







g_qsb_jit_hook = [] {
cudaError_t rc = cudaFuncSetAttribute(
kernel_digest, cudaFuncAttributePreferredSharedMemoryCarveout,
cudaSharedmemCarveoutMaxShared);
if (rc != cudaSuccess) {
fprintf(stderr, "WARN: digest shared-memory carveout hint unavailable: %s\n",
cudaGetErrorString(rc));
(void)cudaGetLastError();
}
};
if (!g_qsb_carrier.on) g_qsb_jit_hook();
cudaError_t qsb_carveout_rc = cudaSuccess;
if (qsb_carrier_has(QK_DIG)) {
qsb_carveout_rc = cudaFuncSetAttribute((const void *)g_qsb_carrier.k[QK_DIG],
cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
if (qsb_carveout_rc != cudaSuccess) {
fprintf(stderr, "WARN: carrier digest shared-memory carveout hint unavailable: %s\n",
cudaGetErrorString(qsb_carveout_rc));
(void)cudaGetLastError();
}
}
clock_gettime(CLOCK_MONOTONIC, &t0);
t_last_report = t0;
uint64_t total_searched = 0;













char summary_path[256];
mkdir("results", 0755);
snprintf(summary_path, sizeof(summary_path),
"results/digest_summary_gpu%d.txt", gpu_index);

FILE *summary_f = fopen(summary_path, "w");
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "STARTED %ld gpu=%d seq=0x%08x lt=%u calibrate=%d easy=%d single_hash=%d\n",
(long)now_epoch, gpu_index, seq_val, lt_val, calibrate, easy, single_hash);
fprintf(summary_f, "# Sanity check: this line proves the file is writable.\n");
fprintf(summary_f, "# Format: STARTED|PROGRESS|HIT|STATUS=...\n");
fprintf(summary_f, "# Hits are also written to digest_hit_<gpu>.txt and digest_calibrate_<gpu>.txt\n");
fflush(summary_f);
QSB_SUMMARY_FSYNC(summary_f);
} else {
fprintf(stderr, "WARN: cannot open summary file %s\n", summary_path);
}
uint64_t hit_counter = 0;




g_summary_f = summary_f;
signal(SIGTERM, on_term_signal);
signal(SIGINT, on_term_signal);
signal(SIGHUP, on_term_signal);




auto binom = [](int n, int k) -> uint64_t {
if (k < 0 || k > n || n < 0) return 0;
if (k > n - k) k = n - k;
uint64_t r = 1;
for (int i = 0; i < k; i++) {
r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
}
return r;
};
uint64_t my_slice_total = 0;
if (tile_path) {

for (int t = 0; t < num_tiles; t++) {
int f = (int)tile_first[t];
int lo = (int)tile_lo[t];
int hi = (int)tile_hi[t];
for (int s = lo; s < hi; s++) {
my_slice_total += binom(n_pool - s - 1, t_sel - 2);
}
}
} else {
for (int f = effective_id; f <= n_pool - t_sel; f += effective_total) {
my_slice_total += binom(n_pool - f - 1, t_sel - 1);
}
}

uint64_t global_total = epoch_mode ? (per_epoch * n_epochs)
: binom(n_pool, t_sel);
printf("  Search space (GLOBAL): C(%d,%d) = %llu combos\n",
n_pool, t_sel, (unsigned long long)global_total);
printf("  Search space (this GPU's slice): %llu combos (%.3f%% of global)\n",
(unsigned long long)my_slice_total,
100.0 * my_slice_total / (double)global_total);
int found = 0;






if (se_mode) {
printf("  Using short-epoch producer/consumer path (%zu epochs per launch)\n",
launch_epochs);
#if QSB_S3
if (!qsb_s3_selfcheck()) {
fprintf(stderr, "ERROR: GLV12 decode tables do not match the geometry\n"); return 1;
}
#elif QSB_DIGIT_SHIFT && !ZLAB_T14
if (gt_shift(2)+1 != 36 || gt_width(2) != 17 || gt_width(13) != 17) {
fprintf(stderr, "ERROR: digit-shift geometry mismatch\n"); return 1;
}
#endif
fflush(stdout);
uint64_t epoch_base = 0;
struct timespec t_last_se = t0;
#if ZLAB_HITPATH


uint8_t *d_hitbuf = NULL;
cudaMalloc(&d_hitbuf, 4 + (size_t)1024 * ZLAB_HIT_REC);
if (!d_hitbuf) { fprintf(stderr, "OOM: hit buffer\n"); return 1; }
uint8_t *d_verified_hitbuf=NULL;
cudaMalloc(&d_verified_hitbuf,4+(size_t)1024*ZLAB_HIT_REC);
if(!d_verified_hitbuf){fprintf(stderr,"OOM: verified hit buffer\n");return 1;}

uint32_t *zh_cnt = (uint32_t *)d_hitbuf;
uint32_t *zh_idx = (uint32_t *)(d_hitbuf + 4);
uint8_t *zh_combos = d_hitbuf + 8;
mkdir("results", 0755);
char zh_fname[256];
if (calibrate) snprintf(zh_fname, sizeof(zh_fname), "results/digest_calibrate_%d.txt", gpu_index);
else snprintf(zh_fname, sizeof(zh_fname), "results/digest_hit_%d.txt", gpu_index);
int zh_fd = open(zh_fname, O_WRONLY | O_CREAT | O_APPEND, 0644);
if (zh_fd < 0) { fprintf(stderr, "ERROR: cannot open %s\n", zh_fname); return 1; }
uint8_t zh_host[4 + 64 * ZLAB_HIT_REC];
#endif
#if QSB_HOST_VERIFY
qsb_hv_t hv;
if (!qsb_hv_init(&hv, &dp, g_hv_win3, window_start, s_early)) { fprintf(stderr, "ERROR: host verify init failed\n"); return 1; }
#if QSB_CPU_GRIND
qcpu::start(&dp, g_hv_win3, QSB_SE_PER_EPOCH, window_start, s_early);
#endif
static uint8_t hv_pend[4 + 1024 * ZLAB_HIT_REC]; uint32_t hv_pend_n = 0; uint64_t hv_pend_base = 0; int hv_pend_epochs = 0;
#endif
#if ZLAB_HITPATH && QSB_SLOT_PIPELINE
#if !QSB_HOST_VERIFY || !QSB_EPOCH_GROUPS
#error "QSB_SLOT_PIPELINE=1 needs QSB_HOST_VERIFY=1 and QSB_EPOCH_GROUPS=1"
#endif







enum { SP_HOST_BYTES = 4 + 256 * ZLAB_HIT_REC };
cudaStream_t sp_stream[2];
cudaEvent_t sp_done[2];
uint8_t *d_hitbuf_s[2] = {d_hitbuf, NULL};
uint8_t *h_tent = NULL;
{
cudaError_t se = cudaSuccess;
for (int s = 0; s < 2 && se == cudaSuccess; s++) {
se = cudaStreamCreateWithFlags(&sp_stream[s], cudaStreamNonBlocking);
if (se == cudaSuccess) se = cudaEventCreateWithFlags(&sp_done[s], cudaEventDisableTiming | (QSB_HOST_BLOCKING ? cudaEventBlockingSync : 0));
}
if (se == cudaSuccess) se = cudaMalloc(&d_hitbuf_s[1], 4 + (size_t)1024 * ZLAB_HIT_REC);
if (se == cudaSuccess) se = cudaHostAlloc((void **)&h_tent, 2 * (size_t)SP_HOST_BYTES, cudaHostAllocDefault);
if (se != cudaSuccess) { fprintf(stderr, "Slot pipeline setup failed: %s\n", cudaGetErrorString(se)); return 1; }
}


{ cudaError_t se = cudaDeviceSynchronize();
if (se == cudaSuccess) se = cudaGetLastError();
if (se != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(se)); return 1; } }
#if QSB_TABLE_L2_WINDOW
qsb_table_l2_window(sp_stream, 2, d_gt, gt_sz);
#endif
int sp_busy[2] = {0, 0};
#if QSB_DIGEST_PHASE_AUDIT
QsbDigestPhaseAudit dp_audit;
uint64_t dp_batch[2]={0,0};
{cudaError_t e=dp_audit.init(d_first_s);if(e!=cudaSuccess){fprintf(stderr,"Digest phase audit init: %s\n",cudaGetErrorString(e));return 1;}}
#endif
#if QSB_PRODUCER_CENSUS
QsbProducerCensus sp_census;
{ cudaError_t ce=sp_census.init(gpu_index, sp_stream);
if(ce!=cudaSuccess) { fprintf(stderr,"Producer census setup: %s\n",cudaGetErrorString(ce)); return 1; } }
auto census_mark = [&](int s, int p) -> int {
cudaError_t ce=sp_census.mark(s,p,sp_stream[s]);
if(ce!=cudaSuccess) { fprintf(stderr,"Producer census mark: %s\n",cudaGetErrorString(ce)); return 1; }
return 0;
};
#endif
#if QSB_FAST_TEARDOWN


static std::atomic<int> ft_done{0}; static double ft_unmap = -2; bool ft_started = false; double ft_pin = -1;
if (sem_init(&g_ft_sem, 0, 0) == 0) {
std::thread([] {
while (sem_wait(&g_ft_sem) != 0) {}
#if QSB_CPU_GRIND && defined(CPU_COUNT)
if (qcpu::g_initial_ok) sched_setaffinity(0, sizeof qcpu::g_initial_cpus, &qcpu::g_initial_cpus);
#endif
#if QSB_CPU_GRIND
ft_unmap = qcpu::stop_unmap();
#endif
ft_done = 1;
}).detach();
g_ft_sem_ok = 1;
} else ft_done = 1;
auto ft_start = [&]() { if (g_stop_signal && !ft_started && g_ft_sem_ok) { ft_started = true; sem_post(&g_ft_sem); } };
#endif
int sp_epochs[2] = {0, 0};
uint64_t sp_base[2] = {0, 0};
uint64_t sp_batch_no = 0;


struct CompletedSubset {
uint8_t bytes[SP_HOST_BYTES];
uint64_t base;
int epochs;
bool valid;
};
auto sp_collect = [&](int s, CompletedSubset &completed) -> int {
completed.valid = false;
if (!sp_busy[s]) return 0;
cudaError_t err = cudaEventSynchronize(sp_done[s]);
if (err == cudaSuccess) err = cudaGetLastError();
if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
#if QSB_PRODUCER_CENSUS
err=sp_census.collect(s);
if(err!=cudaSuccess) { fprintf(stderr,"Producer census collect: %s\n",cudaGetErrorString(err)); return 1; }
#endif

#if QSB_DIGEST_PHASE_AUDIT
err=dp_audit.collect(s,dp_batch[s],sp_epochs[s]);
if(err!=cudaSuccess){fprintf(stderr,"Digest phase collect: %s\n",cudaGetErrorString(err));return 1;}
#endif
completed.base = sp_base[s];
completed.epochs = sp_epochs[s];
memcpy(completed.bytes, h_tent + (size_t)s * SP_HOST_BYTES, SP_HOST_BYTES);
completed.valid = true;
sp_busy[s] = 0;
total_searched += (uint64_t)completed.epochs * QSB_SE_PER_EPOCH;
g_total_searched = total_searched;
return 0;
};
auto sp_publish = [&](const CompletedSubset &completed) -> int {
if (!completed.valid) return 0;
const uint8_t *zh = completed.bytes;
uint32_t nt; memcpy(&nt, zh, 4);
const uint32_t cap = (uint32_t)((SP_HOST_BYTES - 4) / ZLAB_HIT_REC);
if (nt > cap) nt = cap;
for (uint32_t i = 0; i < nt; i++) {
uint32_t tag; memcpy(&tag, zh + 4 + (size_t)i * ZLAB_HIT_REC, 4);
const uint32_t index = tag & 0x3fffffffu, ep = index / (uint32_t)QSB_SE_PER_EPOCH, lane = index % (uint32_t)QSB_SE_PER_EPOCH;
if (ep >= (uint32_t)completed.epochs) continue;
if (qsb_hv_publish(&hv, completed.base + ep, lane, (int)((tag >> 30) & 1u), zh_fd, &hit_counter) < 0) {
fprintf(stderr, "ERROR: hit write failed\n"); return 1;
}
}
g_hit_counter = hit_counter;
return 0;
};
auto sp_drain = [&](int s) -> int {
CompletedSubset completed;
if (sp_collect(s, completed)) return 1;
return sp_publish(completed);
};
auto sp_launch = [&](int s, uint64_t base, int epochs_in_batch) -> int {
cudaStream_t st = sp_stream[s];
epoch_desc_t *d_ep = d_epochs_s[s];
uint32_t *d_fi = d_first_s[s];
uint8_t *d_hb = d_hitbuf_s[s];
uint32_t *cnt = (uint32_t *)d_hb;
uint32_t *idx = (uint32_t *)(d_hb + 4);
uint8_t *combos = d_hb + 8;
const int nblk = ((epochs_in_batch + QSB_PAIR_MUL - 1) / QSB_PAIR_MUL) * QSB_CTA_PARTS;
const int batch_pos = nblk * QSB_SE_BLOCK;
#if QSB_PRODUCER_CENSUS
sp_census.start(s,sp_batch_no,epochs_in_batch);
#endif
#if QSB_TRACE_START
if ((sp_batch_no & (sp_batch_no - 1)) == 0 && sp_batch_no <= 64)
fprintf(stderr, "TRACE launch %llu at %.1f ms\n", (unsigned long long)sp_batch_no, (qsb_trace_now() - qsb_trace_t_start) * 1e3);
#endif
#ifdef QSB_HP_ON


#if QSB_HP_V3
#if QSB_PRODUCER_CENSUS
double census_check_start=QsbProducerCensus::now_ms();
#endif
qhp::wait_check_reuse(d_ep);
#if QSB_PRODUCER_CENSUS
sp_census.check_ms[s]=QsbProducerCensus::now_ms()-census_check_start;
#endif
#endif
qhp::poll();
#if QSB_PRODUCER_CENSUS
double census_acquire_start=QsbProducerCensus::now_ms();
#endif
qhp::Slot *hp_slot = qhp::acquire((int64_t)sp_batch_no);
#if QSB_PRODUCER_CENSUS
sp_census.acquire_ms[s]=QsbProducerCensus::now_ms()-census_acquire_start;
#endif
if (hp_slot) {
#if QSB_PRODUCER_CENSUS
sp_census.host[s]=1;
if(census_mark(s,0) || census_mark(s,1)) return 1;
#endif


cudaError_t he = cudaMemsetAsync(cnt, 0, sizeof(uint32_t), st);
if (he == cudaSuccess) he = qhp::upload(hp_slot, st, d_ep, d_fi, (size_t)QSB_FIRST_SLOTS * 8 * sizeof(uint32_t), epochs_in_batch
#if QSB_HP_V3
, QSB_HP_FIRST_ONLY_UPLOAD != 0
#if QSB_FIRST_LOCAL || QSB_FIRST_DEVICE
, true
#endif
#endif
);
if (he != cudaSuccess) { printf("CUDA error: %s (host producer upload)\n", cudaGetErrorString(he)); return 1; }
#if QSB_FIRST_DEVICE
{ const unsigned nthr = (unsigned)epochs_in_batch * (unsigned)qsb_first_class_count;
if (!qsb_carrier_try(kernel_build_first_flat, QK_BFF, dim3((nthr + 255) / 256), dim3(256), st,
d_ep, d_fi, (unsigned)epochs_in_batch, (unsigned)qsb_first_class_count))
kernel_build_first_flat<<<(nthr + 255) / 256, 256, 0, st>>>(d_ep, d_fi, (unsigned)epochs_in_batch, (unsigned)qsb_first_class_count); }
#endif
} else
#endif
{{
#if QSB_PRODUCER_CENSUS
if(census_mark(s,0)) return 1;
#endif
uint8_t h_o[MAX_T];
qsb_host_unrank(base, window_start, s_early, h_o);
const uint64_t r5a = qsb_host_rank(h_o, s_early - 1, window_start);
qsb_host_unrank(base + (uint64_t)epochs_in_batch - 1, window_start, s_early, h_o);
const uint64_t r5b = qsb_host_rank(h_o, s_early - 1, window_start);
const uint64_t n_groups64 = r5b - r5a + 1;
const uint32_t n_groups = (uint32_t)n_groups64;
if (n_groups64 > (uint64_t)group_capacity) {
#if QSB_TRIM_DIRECT_PRODUCER
fprintf(stderr, "ERROR: epoch-group capacity exceeded (%llu groups); the direct producer is compiled out\n", (unsigned long long)n_groups64); return 1;
#else
kernel_build_epochs<<<(epochs_in_batch + 255) / 256, 256, 0, st>>>(
base, base + epochs_in_batch, window_start, s_early,
d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_ep, cnt);
#endif
} else {



if (!qsb_carrier_try(kernel_epoch_groups, QK_EG, dim3((n_groups + 255) / 256), dim3(256), st,
r5a, n_groups, window_start, s_early, d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_groups_s[s]
#if QSB_EPOCH_FAST
, d_epoch_group_s[s], base, base + (uint64_t)epochs_in_batch
#endif
))
kernel_epoch_groups<<<(n_groups + 255) / 256, 256, 0, st>>>(
r5a, n_groups, window_start, s_early, d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_groups_s[s]
#if QSB_EPOCH_FAST
, d_epoch_group_s[s], base, base + (uint64_t)epochs_in_batch
#endif
);
if (!qsb_carrier_try(kernel_build_epochs_inc, QK_BEI, dim3((epochs_in_batch + 255) / 256), dim3(256), st,
base, base + epochs_in_batch, window_start, s_early,
d_dsigs, d_groups_s[s], r5a, d_ep, cnt
#if QSB_EPOCH_FAST
, d_epoch_group_s[s]
#endif
))
kernel_build_epochs_inc<<<(epochs_in_batch + 255) / 256, 256, 0, st>>>(
base, base + epochs_in_batch, window_start, s_early,
d_dsigs, d_groups_s[s], r5a, d_ep, cnt
#if QSB_EPOCH_FAST
, d_epoch_group_s[s]
#endif
);
}
}
#if QSB_PRODUCER_CENSUS
if(census_mark(s,1)) return 1;
#endif
#if QSB_FIRST_LOCAL
/* Keep batch0 independent whole-array check; steady fallback uses descriptors. */
if(sp_batch_no==0)
#endif
{ const unsigned nthr = (unsigned)epochs_in_batch * (unsigned)qsb_first_class_count;
if (!qsb_carrier_try(kernel_build_first_flat, QK_BFF, dim3((nthr + 255) / 256), dim3(256), st,
d_ep, d_fi, (unsigned)epochs_in_batch, (unsigned)qsb_first_class_count))
kernel_build_first_flat<<<(nthr + 255) / 256, 256, 0, st>>>(d_ep, d_fi, (unsigned)epochs_in_batch, (unsigned)qsb_first_class_count); }
#ifdef QSB_HP_ON
if (sp_batch_no == 0)
qhp::enqueue_check_copy(st, d_ep, d_fi, (size_t)QSB_FIRST_SLOTS * 8 * sizeof(uint32_t));
#endif
}
#if QSB_PRODUCER_CENSUS
if(census_mark(s,2)) return 1;
#endif

#if QSB_DIGEST_PHASE_AUDIT
{cudaError_t e=dp_audit.clear(s,st);if(e!=cudaSuccess){fprintf(stderr,"Digest phase clear: %s\n",cudaGetErrorString(e));return 1;}dp_batch[s]=sp_batch_no;}
#endif
if (!qsb_carrier_try(kernel_digest, QK_DIG, dim3(nblk), dim3(QSB_SE_BLOCK), st,
(const uint8_t*)NULL, n_pool, t_sel,
d_mid,
d_prem, 0,
d_dsigs, d_tail, dp.tail_section_len,
d_suf, dp.tx_suffix_len, dp.total_preimage_len,
d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
d_gt,
cnt, idx,
combos, d_hit_sighash,
d_hit_keynonce, d_hit_pubhash,
d_hit_qx, d_hit_qy,
batch_pos, easy, single_hash, calibrate, window_start, (uint64_t)0,
t_win, s_early, d_early, fast_inc, d_const_words, d_ep, d_fi, epochs_in_batch))
kernel_digest<<<nblk, QSB_SE_BLOCK, 0, st>>>(
(const uint8_t*)NULL, n_pool, t_sel,
d_mid,
d_prem, 0,
d_dsigs, d_tail, dp.tail_section_len,
d_suf, dp.tx_suffix_len, dp.total_preimage_len,
d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
d_gt,
cnt, idx,
combos, d_hit_sighash,
d_hit_keynonce, d_hit_pubhash,
d_hit_qx, d_hit_qy,
batch_pos, easy, single_hash, calibrate, window_start, (uint64_t)0,
t_win, s_early, d_early, fast_inc, d_const_words, d_ep, d_fi, epochs_in_batch);
#if QSB_PRODUCER_CENSUS
if(census_mark(s,3)) return 1;
#endif
sp_base[s] = base;
cudaError_t err = cudaMemcpyAsync(h_tent + (size_t)s * SP_HOST_BYTES, d_hb, SP_HOST_BYTES, cudaMemcpyDeviceToHost, st);
if (err == cudaSuccess) err = cudaEventRecord(sp_done[s], st);
if (err == cudaSuccess) err = cudaGetLastError();
if (err != cudaSuccess) { printf("CUDA error: %s (batch %llu enqueue)\n", cudaGetErrorString(err), (unsigned long long)sp_batch_no); return 1; }

sp_busy[s] = 1;
sp_epochs[s] = epochs_in_batch;
return 0;
};
g_stop_polled = 1;
g_qsb_carrier.running = 1;
while (1) {
const int s = (int)(sp_batch_no & 1);
#if QSB_SP_REFILL_FIRST
CompletedSubset completed;
if (sp_collect(s, completed)) return 1;
if (g_stop_signal || epoch_base >= n_epochs) {
#if QSB_FAST_TEARDOWN
ft_start();
#endif
if (sp_publish(completed)) return 1;
if (sp_drain(s ^ 1)) return 1;
break;
}
#else
if (sp_drain(s)) return 1;
if (g_stop_signal || epoch_base >= n_epochs) {
#if QSB_FAST_TEARDOWN
ft_start();
#endif
if (sp_drain(s ^ 1)) return 1;
break;
}
#endif
const uint64_t epochs_left = n_epochs - epoch_base;
const uint64_t capacity = (uint64_t)launch_epochs;
const int epochs_in_batch = (int)(epochs_left < capacity ? epochs_left : capacity);
#if QSB_SP_REFILL_FIRST
const int launch_error = sp_launch(s, epoch_base, epochs_in_batch);
if (sp_publish(completed)) return 1;
if (launch_error) return 1;
#else
if (sp_launch(s, epoch_base, epochs_in_batch)) return 1;
#endif
epoch_base += epochs_in_batch;
sp_batch_no++;
struct timespec t_now;
clock_gettime(CLOCK_MONOTONIC, &t_now);
double secs_since = (t_now.tv_sec - t_last_se.tv_sec)
+ (t_now.tv_nsec - t_last_se.tv_nsec) / 1e9;
if (secs_since >= 15.0) {
double elapsed_total = (t_now.tv_sec - t0.tv_sec)
+ (t_now.tv_nsec - t0.tv_nsec) / 1e9;
double rate = total_searched / elapsed_total;
printf("  [GPU %d] epoch=%llu/%llu (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs\n",
gpu_index,
(unsigned long long)epoch_base, (unsigned long long)n_epochs,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(global_total/1000000),
rate/1e6, elapsed_total);
#ifdef QSB_HP_ON
{ uint64_t hb, fb; int hst; qhp::stats(&hb, &fb, &hst);
if (hst != -2) printf("  [HP] host-built batches %llu, GPU-built after start-up %llu, host producers %s\n",
(unsigned long long)hb, (unsigned long long)fb, hst == 1 ? "on" : hst == 0 ? "pending" : "off"); }
#endif
fflush(stdout);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "PROGRESS %ld attempts=%llu rate_M_per_s=%.1f elapsed_s=%.0f hits_so_far=%llu\n",
(long)now_epoch, (unsigned long long)total_searched,
rate/1e6, elapsed_total, (unsigned long long)hit_counter);
fflush(summary_f);
}
t_last_se = t_now;
}
}
#ifdef QSB_HP_ON
qhp::shutdown();
#if QSB_FAST_TEARDOWN && QSB_FAST_TEARDOWN_PIN
if (g_stop_signal && qhp::g_hp) {
struct timespec fa, fb; clock_gettime(CLOCK_MONOTONIC, &fa);
for (auto &sl : qhp::g_hp->slot) {
for (int p = 0; p < sl.npieces_ok; p++) { cudaFreeHost(sl.ep[p]); if(sl.fi[p]) cudaFreeHost(sl.fi[p]); sl.ep[p] = nullptr; sl.fi[p] = nullptr; }
sl.npieces_ok = 0;
}
(void)cudaGetLastError();
clock_gettime(CLOCK_MONOTONIC, &fb); ft_pin = (fb.tv_sec - fa.tv_sec) + 1e-9 * (fb.tv_nsec - fa.tv_nsec);
}
#endif
{ uint64_t hb, fb; int hst, amin; double aavg; qhp::stats(&hb, &fb, &hst, &aavg, &amin);
if (hst != -2) printf("  [HP] final: host-built batches %llu, GPU-built after start-up %llu (of %llu); ready ahead at launch: avg %.2f, min %d\n",
(unsigned long long)hb, (unsigned long long)fb, (unsigned long long)sp_batch_no, aavg, amin); }
#endif
#if QSB_PRODUCER_CENSUS
{ cudaError_t ce=sp_census.finish();
if(ce!=cudaSuccess) { fprintf(stderr,"Producer census finish: %s\n",cudaGetErrorString(ce)); return 1; } }
#endif
g_stop_polled = 0;
#else
#if QSB_TABLE_L2_WINDOW
{ cudaStream_t legacy = cudaStreamLegacy; qsb_table_l2_window(&legacy, 1, d_gt, gt_sz); }
#endif
while (1) {
uint64_t epochs_left = n_epochs - epoch_base;
const uint64_t capacity=(uint64_t)launch_epochs;
const int epochs_in_batch=(int)(epochs_left<capacity?epochs_left:capacity);
int nblk=(epochs_in_batch+QSB_PAIR_MUL-1)/QSB_PAIR_MUL;
int batch_pos = nblk * QSB_SE_BLOCK;
uint32_t h_hit = 0;
#if ZLAB_HITPATH && QSB_EPOCH_GROUPS
{
uint8_t h_o[MAX_T];
qsb_host_unrank(epoch_base, window_start, s_early, h_o);
const uint64_t r5a = qsb_host_rank(h_o, s_early - 1, window_start);
qsb_host_unrank(epoch_base + (uint64_t)epochs_in_batch - 1, window_start, s_early, h_o);
const uint64_t r5b = qsb_host_rank(h_o, s_early - 1, window_start);
const uint64_t n_groups64 = r5b - r5a + 1;
const uint32_t n_groups = (uint32_t)n_groups64;
if (n_groups64 > (uint64_t)group_capacity) {
#if QSB_TRIM_DIRECT_PRODUCER
fprintf(stderr, "ERROR: epoch-group capacity exceeded (%llu groups); the direct producer is compiled out\n", (unsigned long long)n_groups64); return 1;
#else

kernel_build_epochs<<<(epochs_in_batch + 255) / 256, 256>>>(
epoch_base, epoch_base+epochs_in_batch, window_start, s_early,
d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_epochs, zh_cnt);
#endif
} else {
kernel_epoch_groups<<<(n_groups + 255) / 256, 256>>>(
r5a, n_groups, window_start, s_early, d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_groups
#if QSB_EPOCH_FAST
, d_epoch_group, epoch_base, epoch_base+(uint64_t)epochs_in_batch
#endif
);
kernel_build_epochs_inc<<<(epochs_in_batch + 255) / 256, 256>>>(
epoch_base, epoch_base+epochs_in_batch, window_start, s_early,
d_dsigs, d_groups, r5a, d_epochs, zh_cnt
#if QSB_EPOCH_FAST
, d_epoch_group
#endif
);
}
}
#elif ZLAB_HITPATH
kernel_build_epochs<<<(epochs_in_batch + 255) / 256, 256>>>(
epoch_base, epoch_base+epochs_in_batch, window_start, s_early,
d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_epochs, zh_cnt);
#else
cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);
kernel_build_epochs<<<(epochs_in_batch + 255) / 256, 256>>>(
epoch_base, epoch_base+epochs_in_batch, window_start, s_early,
d_mid, d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_epochs);
#endif

{ const unsigned nthr=(unsigned)epochs_in_batch*(unsigned)qsb_first_class_count;
kernel_build_first_flat<<<(nthr+255)/256,256>>>(d_epochs,d_first,(unsigned)epochs_in_batch,(unsigned)qsb_first_class_count); }
kernel_digest<<<nblk, QSB_SE_BLOCK>>>(
(const uint8_t*)NULL, n_pool, t_sel,
d_mid,
d_prem, 0,
d_dsigs, d_tail, dp.tail_section_len,
d_suf, dp.tx_suffix_len, dp.total_preimage_len,
d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
d_gt,
#if ZLAB_HITPATH
zh_cnt, zh_idx,
zh_combos, d_hit_sighash,
#else
d_hit_cnt, d_hit_idx,
d_hit_combos, d_hit_sighash,
#endif
d_hit_keynonce, d_hit_pubhash,
d_hit_qx, d_hit_qy,
batch_pos, easy, single_hash, calibrate, window_start, (uint64_t)0,
t_win, s_early, d_early, fast_inc, d_const_words, d_epochs, d_first, epochs_in_batch);
#if QSB_HOST_VERIFY

for (uint32_t i = 0; i < hv_pend_n; i++) {
uint32_t tag; memcpy(&tag, hv_pend + 4 + (size_t)i * ZLAB_HIT_REC, 4);
const uint32_t index = tag & 0x3fffffffu, ep = index / (uint32_t)QSB_SE_PER_EPOCH, lane = index % (uint32_t)QSB_SE_PER_EPOCH;
if (ep >= (uint32_t)hv_pend_epochs) continue;
int r = qsb_hv_publish(&hv, hv_pend_base + ep, lane, (int)((tag >> 30) & 1u), zh_fd, &hit_counter);
if (r < 0) { fprintf(stderr, "ERROR: hit write failed\n"); return 1; }
}
#if QSB_HV_STATS
fprintf(stderr, "hv: tentatives=%u published_total=%llu\n", hv_pend_n, (unsigned long long)hit_counter);
#endif
hv_pend_n = 0; g_hit_counter = hit_counter;
#else
kernel_verify_pair_hits<<<1,64>>>(d_hitbuf,d_verified_hitbuf,d_epochs,d_first,d_gt,epochs_in_batch);
#endif

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
total_searched += (uint64_t)epochs_in_batch*QSB_SE_PER_EPOCH;
epoch_base += epochs_in_batch;
#if ZLAB_HITPATH
#if QSB_HOST_VERIFY
err = cudaMemcpy(hv_pend, d_hitbuf, 4, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) { fprintf(stderr, "Hit read failed: %s\n", cudaGetErrorString(err)); return 1; }
g_total_searched = total_searched;
memcpy(&hv_pend_n, hv_pend, 4);
if (hv_pend_n > 1024u) hv_pend_n = 1024u;
if (hv_pend_n) { err = cudaMemcpy(hv_pend + 4, d_hitbuf + 4, (size_t)hv_pend_n * ZLAB_HIT_REC, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) { fprintf(stderr, "Hit read failed: %s\n", cudaGetErrorString(err)); return 1; } }
hv_pend_base = epoch_base - (uint64_t)epochs_in_batch; hv_pend_epochs = epochs_in_batch;
h_hit = 0;
#else
err = cudaMemcpy(zh_host, d_verified_hitbuf, 4 + ZLAB_HIT_FIRST * ZLAB_HIT_REC, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) { fprintf(stderr, "Hit read failed: %s\n", cudaGetErrorString(err)); return 1; }

g_total_searched = total_searched;
memcpy(&h_hit, zh_host, 4);
#endif
if (h_hit > 0) {
int nh = (h_hit > 64) ? 64 : (int)h_hit;
if (nh > ZLAB_HIT_FIRST)
cudaMemcpy(zh_host + 4 + ZLAB_HIT_FIRST * ZLAB_HIT_REC,
d_verified_hitbuf + 4 + ZLAB_HIT_FIRST * ZLAB_HIT_REC,
(size_t)(nh - ZLAB_HIT_FIRST) * ZLAB_HIT_REC, cudaMemcpyDeviceToHost);

char wb[64 * 96];
int wl = 0;
for (int h = 0; h < nh; h++) {
uint32_t raw; memcpy(&raw, zh_host + 4 + h * ZLAB_HIT_REC, 4);
const uint8_t *combo = zh_host + 8 + h * ZLAB_HIT_REC;


wl += snprintf(wb + wl, sizeof(wb) - wl, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], combo[6], combo[7], combo[8],
(int)((raw >> 30) & 1));
}
const char *wp = wb;
while (wl > 0) {
ssize_t k = write(zh_fd, wp, (size_t)wl);
if (k < 0) { if (errno == EINTR) continue; fprintf(stderr, "ERROR: hit write failed\n"); return 1; }
wp += k; wl -= (int)k;
}
hit_counter += (uint64_t)nh;
g_hit_counter = hit_counter;
}
if (0) {
#else
cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
g_total_searched = total_searched;
if (h_hit > 0) {
#endif
uint32_t hits[64];
int nh = (h_hit > 64) ? 64 : h_hit;
cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);
printf("\n  *** DIGEST HIT! ***\n");
mkdir("results", 0755);
char fname[256];
if (calibrate) snprintf(fname, sizeof(fname), "results/digest_calibrate_%d.txt", gpu_index);
else snprintf(fname, sizeof(fname), "results/digest_hit_%d.txt", gpu_index);
FILE *ff = fopen(fname, "a");
if (ff) {
uint8_t all_combos[1024 * MAX_T];
cudaMemcpy(all_combos, d_hit_combos, nh * MAX_T, cudaMemcpyDeviceToHost);
for (int h = 0; h < nh; h++) {
uint32_t raw = hits[h];
int combo_idx = raw & 0x3FFFFFFF;
int ri = (raw >> 30) & 1;
int hc = (raw >> 31) & 1;
uint8_t *combo = all_combos + h * MAX_T;
fprintf(ff, "indices=");
printf("  indices=");
for (int j = 0; j < t_sel; j++) {
fprintf(ff, "%s%d", j?",":"", combo[j]);
printf("%s%d", j?",":"", combo[j]);
}


fprintf(ff, "\nhash_choice=%d\nrecid=%d\ncombo_idx=%d\n", hc, ri, combo_idx);
printf(" hc=%d recid=%d\n", hc, ri);
hit_counter++;
g_hit_counter = hit_counter;
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "HIT %ld combo=", (long)now_epoch);
for (int j = 0; j < t_sel; j++)
fprintf(summary_f, "%s%d", j?",":"", combo[j]);
fprintf(summary_f, " hash_choice=%d recid=%d", hc, ri);
fprintf(summary_f, " combo_idx=%d calibrate=%d\n", combo_idx, calibrate);
fflush(summary_f);

}
}
fclose(ff);
}
}
struct timespec t_now;
clock_gettime(CLOCK_MONOTONIC, &t_now);
double secs_since = (t_now.tv_sec - t_last_se.tv_sec)
+ (t_now.tv_nsec - t_last_se.tv_nsec) / 1e9;
if (secs_since >= 15.0) {
double elapsed_total = (t_now.tv_sec - t0.tv_sec)
+ (t_now.tv_nsec - t0.tv_nsec) / 1e9;
double rate = total_searched / elapsed_total;
printf("  [GPU %d] epoch=%llu/%llu (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs\n",
gpu_index,
(unsigned long long)epoch_base, (unsigned long long)n_epochs,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(global_total/1000000),
rate/1e6, elapsed_total);
fflush(stdout);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "PROGRESS %ld attempts=%llu rate_M_per_s=%.1f elapsed_s=%.0f hits_so_far=%llu\n",
(long)now_epoch, (unsigned long long)total_searched,
rate/1e6, elapsed_total, (unsigned long long)hit_counter);
fflush(summary_f);
}
t_last_se = t_now;
}
#if QSB_HOST_VERIFY
if (epoch_base >= n_epochs) {
for (uint32_t i = 0; i < hv_pend_n; i++) {
uint32_t tag; memcpy(&tag, hv_pend + 4 + (size_t)i * ZLAB_HIT_REC, 4);
const uint32_t index = tag & 0x3fffffffu, ep = index / (uint32_t)QSB_SE_PER_EPOCH, lane = index % (uint32_t)QSB_SE_PER_EPOCH;
if (ep >= (uint32_t)hv_pend_epochs) continue;
if (qsb_hv_publish(&hv, hv_pend_base + ep, lane, (int)((tag >> 30) & 1u), zh_fd, &hit_counter) < 0) { fprintf(stderr, "ERROR: hit write failed\n"); return 1; }
}
hv_pend_n = 0; g_hit_counter = hit_counter;
}
#endif
if (epoch_base >= n_epochs) break;
}
#endif
clock_gettime(CLOCK_MONOTONIC, &t1);
double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
#if ZLAB_HITPATH && QSB_SLOT_PIPELINE
if (g_stop_signal) {
#if QSB_FAST_TEARDOWN
{ struct timespec fw0, fw; clock_gettime(CLOCK_MONOTONIC, &fw0);
do { if (ft_done.load()) break; usleep(200); clock_gettime(CLOCK_MONOTONIC, &fw); } while ((fw.tv_sec - fw0.tv_sec) + 1e-9 * (fw.tv_nsec - fw0.tv_nsec) < 3.0);
clock_gettime(CLOCK_MONOTONIC, &fw);
printf("  Teardown: co-grinder table %s %.3f s after the stop signal; pinned slots freed in %.3f s; waited %.3f s after the drain\n",
ft_unmap >= 0 ? "unmapped" : "left to the exit,", ft_unmap >= 0 ? ft_unmap : 0.0, ft_pin > 0 ? ft_pin : 0.0,
(fw.tv_sec - fw0.tv_sec) + 1e-9 * (fw.tv_nsec - fw0.tv_nsec)); }
#endif



printf("  [GPU %d] epoch=%llu/%llu (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs\n",
gpu_index, (unsigned long long)epoch_base, (unsigned long long)n_epochs,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(global_total/1000000),
total_searched/elapsed/1e6, elapsed);
#if QSB_CPU_GRIND
if (qcpu::g_ctx) printf("  CPU co-grind: %lluM candidates, %u hits (%.2fM/s)\n",
(unsigned long long)(qcpu::candidates() / 1000000), qcpu::hits(),
qcpu::candidates() / elapsed / 1e6);
#endif
printf("\n  [GPU %d] Stopped by signal %d after draining every launched batch: %lluM in %.0fs (%.1fM/s)\n",
gpu_index, (int)g_stop_signal, (unsigned long long)(total_searched/1000000), elapsed,
total_searched/elapsed/1e6);
fflush(stdout);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "STATUS=KILLED %ld signal=%d total_attempts=%llu hits=%llu\n",
(long)now_epoch, (int)g_stop_signal, (unsigned long long)total_searched,
(unsigned long long)hit_counter);
fflush(summary_f); QSB_SUMMARY_FSYNC(summary_f); fclose(summary_f);
g_summary_f = NULL;
}
free(h_combos);
#if QSB_CPU_GRIND


if (qcpu::g_ctx) {
qcpu::g_ctx->io.lock();
if (qcpu::g_ctx->out) fflush(qcpu::g_ctx->out);
fflush(NULL);
_exit(0);
}
#endif
return 0;
}
#endif
printf("\n  [GPU %d] Done short-epoch: %lluM in %.0fs (%.1fM/s)\n", gpu_index,
(unsigned long long)(total_searched/1000000), elapsed,
total_searched/elapsed/1e6);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "STATUS=EXHAUSTED %ld total_attempts=%llu elapsed_s=%.0f hits=%llu\n",
(long)now_epoch, (unsigned long long)total_searched,
elapsed, (unsigned long long)hit_counter);
fflush(summary_f); QSB_SUMMARY_FSYNC(summary_f); fclose(summary_f);
g_summary_f = NULL;
}
free(h_combos);
return 0;
}
#if ZLAB_TRIM
fprintf(stderr, "ERROR: ZLAB_TRIM build supports only the ranked short-epoch shape\n");
return 1;
#else




if (tile_path == NULL && effective_total == 1) {
printf("  Using GPU-enum fast path (no CPU fill, base-linear)\n");
fflush(stdout);
uint64_t enum_base = 0;
uint64_t epoch = 0;
uint64_t span = epoch_mode ? per_epoch : global_total;
int prem_len_now = (int)dp.prefix_remainder_len;
int need_epoch = epoch_mode;
struct timespec t_last_enum = t0;
while (!found) {
if (need_epoch) {


unrank_combo_host(epoch, window_start, s_early, epoch_skip);
build_epoch_prefix(&dp, window_start, s_early, epoch_skip, epoch_prefix,
epoch_mid, epoch_rem, &epoch_rem_len);
cudaMemcpy(d_mid, epoch_mid, 32, cudaMemcpyHostToDevice);
if (epoch_rem_len > 0)
cudaMemcpy(d_prem, epoch_rem, epoch_rem_len, cudaMemcpyHostToDevice);
cudaMemcpy(d_early, epoch_skip, s_early, cudaMemcpyHostToDevice);
prem_len_now = epoch_rem_len;
need_epoch = 0;
}
int batch_pos = (int)((span - enum_base < (uint64_t)BATCH) ? span - enum_base : BATCH);
uint32_t h_hit = 0;
cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);
int grdsz = (batch_pos + BLKSZ - 1) / BLKSZ;
#if defined(QSB_PAIR_SHARED) && !QSB_PAIR_SHARED
if(qsb_prefix_eligible(n_pool,window_start,t_win,fast_inc,prem_len_now))
qsb_prepare_prefix_cache<<<(QSB_PREFIX_ENTRIES+255)/256,256>>>(d_mid,window_start,t_win);
#endif
kernel_digest<<<grdsz, BLKSZ>>>(
(const uint8_t*)NULL, n_pool, t_sel,
d_mid,
d_prem, prem_len_now,
d_dsigs, d_tail, dp.tail_section_len,
d_suf, dp.tx_suffix_len, dp.total_preimage_len,
d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
d_gt,
d_hit_cnt, d_hit_idx,
d_hit_combos, d_hit_sighash,
d_hit_keynonce, d_hit_pubhash,
d_hit_qx, d_hit_qy,
batch_pos, easy, single_hash, calibrate, window_start, enum_base,
t_win, s_early, d_early, fast_inc, d_const_words, NULL, NULL, 0);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
total_searched += batch_pos;
enum_base += batch_pos;
err = cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) { fprintf(stderr, "Hit read failed: %s\n", cudaGetErrorString(err)); return 1; }

g_total_searched = total_searched;
if (h_hit > 0) {
uint32_t hits[64];
int nh = (h_hit > 64) ? 64 : h_hit;
cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);
printf("\n  *** DIGEST HIT! ***\n");
mkdir("results", 0755);
char fname[256];
if (calibrate) snprintf(fname, sizeof(fname), "results/digest_calibrate_%d.txt", gpu_index);
else snprintf(fname, sizeof(fname), "results/digest_hit_%d.txt", gpu_index);
FILE *ff = fopen(fname, "a");
if (ff) {
uint8_t all_combos[1024 * MAX_T];
cudaMemcpy(all_combos, d_hit_combos, nh * MAX_T, cudaMemcpyDeviceToHost);
for (int h = 0; h < nh; h++) {
uint32_t raw = hits[h];
int combo_idx = raw & 0x3FFFFFFF;
int ri = (raw >> 30) & 1;
int hc = (raw >> 31) & 1;
uint8_t *combo = all_combos + h * MAX_T;
fprintf(ff, "indices=");
printf("  indices=");
for (int j = 0; j < t_sel; j++) {
fprintf(ff, "%s%d", j?",":"", combo[j]);
printf("%s%d", j?",":"", combo[j]);
}


fprintf(ff, "\nhash_choice=%d\nrecid=%d\ncombo_idx=%d\n", hc, ri, combo_idx);
printf(" hc=%d recid=%d\n", hc, ri);
hit_counter++;
g_hit_counter = hit_counter;
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "HIT %ld combo=", (long)now_epoch);
for (int j = 0; j < t_sel; j++)
fprintf(summary_f, "%s%d", j?",":"", combo[j]);
fprintf(summary_f, " hash_choice=%d recid=%d", hc, ri);
fprintf(summary_f, " combo_idx=%d calibrate=%d\n", combo_idx, calibrate);
fflush(summary_f);

}
}
fclose(ff);
}
}
struct timespec t_now;
clock_gettime(CLOCK_MONOTONIC, &t_now);
double secs_since = (t_now.tv_sec - t_last_enum.tv_sec)
+ (t_now.tv_nsec - t_last_enum.tv_nsec) / 1e9;
if (secs_since >= 15.0) {
double elapsed_total = (t_now.tv_sec - t0.tv_sec)
+ (t_now.tv_nsec - t0.tv_nsec) / 1e9;
double rate = total_searched / elapsed_total;
printf("  [GPU %d] enum_base=%llu (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs\n",
gpu_index, (unsigned long long)enum_base,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(global_total/1000000),
rate/1e6, elapsed_total);
fflush(stdout);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "PROGRESS %ld attempts=%llu rate_M_per_s=%.1f elapsed_s=%.0f hits_so_far=%llu\n",
(long)now_epoch, (unsigned long long)total_searched,
rate/1e6, elapsed_total, (unsigned long long)hit_counter);
fflush(summary_f);
}
t_last_enum = t_now;
}
if (enum_base >= span) {
if (!epoch_mode) break;
enum_base = 0;
epoch++;
need_epoch = 1;
if (epoch >= n_epochs) {







if (s_early + 1 < t_sel) {
s_early += 1;
t_win = t_sel - s_early;


fast_inc = 0;
per_epoch = binom_u64(n_pool - window_start, t_win);
n_epochs = binom_u64(window_start, s_early);
span = per_epoch;
epoch = 0;
global_total += per_epoch * n_epochs;
printf("  Family exhausted; advancing to %d fixed early omissions "
"(%llu epochs x %llu per epoch)\n",
s_early, (unsigned long long)n_epochs,
(unsigned long long)per_epoch);
fflush(stdout);
} else {
break;
}
}
}
}
clock_gettime(CLOCK_MONOTONIC, &t1);
double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
printf("\n  [GPU %d] Done enum: %lluM in %.0fs (%.1fM/s)\n", gpu_index,
(unsigned long long)(total_searched/1000000), elapsed,
total_searched/elapsed/1e6);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f, "STATUS=EXHAUSTED %ld total_attempts=%llu elapsed_s=%.0f hits=%llu\n",
(long)now_epoch, (unsigned long long)total_searched,
elapsed, (unsigned long long)hit_counter);
fflush(summary_f); fsync(fileno(summary_f)); fclose(summary_f);
g_summary_f = NULL;
}
free(h_combos);
return 0;
}





int tile_idx = 0;
int first;
int second_lo, second_hi;
while (!found) {
if (tile_path) {
if (tile_idx >= num_tiles) break;
first = (int)tile_first[tile_idx];
second_lo = (int)tile_lo[tile_idx];
second_hi = (int)tile_hi[tile_idx];
tile_idx++;
} else {

if (tile_idx == 0) {
first = effective_id;
} else {
first += effective_total;
}
tile_idx++;
if (first > n_pool - t_sel) break;
second_lo = first + 1;
second_hi = n_pool - t_sel + 2;
}


int sub[MAX_T];
sub[0] = second_lo;
for (int i = 1; i < t_sel - 1; i++) sub[i] = sub[i-1] + 1;
int batch_pos = 0;
int exhausted = 0;

while (!exhausted && !found) {

while (batch_pos < BATCH && !exhausted) {

if (sub[0] >= second_hi) { exhausted = 1; break; }
h_combos[batch_pos * t_sel] = (uint8_t)first;
for (int i = 0; i < t_sel - 1; i++)
h_combos[batch_pos * t_sel + 1 + i] = (uint8_t)sub[i];
batch_pos++;


int i = t_sel - 2;
while (i >= 0 && sub[i] == n_pool - (t_sel - 1) + i) i--;
if (i < 0) { exhausted = 1; break; }
sub[i]++;
for (int j = i + 1; j < t_sel - 1; j++) sub[j] = sub[j-1] + 1;
}
if (batch_pos == 0) break;


cudaMemcpy(d_combos, h_combos, batch_pos * t_sel, cudaMemcpyHostToDevice);
uint32_t h_hit = 0;
cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);

int grdsz = (batch_pos + BLKSZ - 1) / BLKSZ;
kernel_digest<<<grdsz, BLKSZ>>>(
d_combos, n_pool, t_sel,
d_mid,
d_prem, (int)dp.prefix_remainder_len,
d_dsigs, d_tail, dp.tail_section_len,
d_suf, dp.tx_suffix_len, dp.total_preimage_len,
d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
d_gt,
d_hit_cnt, d_hit_idx,
d_hit_combos, d_hit_sighash,
d_hit_keynonce, d_hit_pubhash,
d_hit_qx, d_hit_qy,
batch_pos, easy, single_hash, calibrate, 0, (uint64_t)0,
t_sel, 0, d_early, 0, d_const_words, NULL, NULL, 0);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

total_searched += batch_pos;
batch_pos = 0;

err = cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) { fprintf(stderr, "Hit read failed: %s\n", cudaGetErrorString(err)); return 1; }

g_total_searched = total_searched;
if (h_hit > 0) {
uint32_t hits[64];
int nh = (h_hit > 64) ? 64 : h_hit;
cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);

printf("\n  *** DIGEST HIT! ***\n");
mkdir("results", 0755);
char fname[256];
if (calibrate) {
snprintf(fname, sizeof(fname), "results/digest_calibrate_%d.txt", gpu_index);
} else {
snprintf(fname, sizeof(fname), "results/digest_hit_%d.txt", gpu_index);
}



FILE *ff = fopen(fname, "a");
if (ff) {
uint8_t all_combos[1024 * MAX_T];
cudaMemcpy(all_combos, d_hit_combos, nh * MAX_T, cudaMemcpyDeviceToHost);

for (int h = 0; h < nh; h++) {
uint32_t raw = hits[h];
int combo_idx = raw & 0x3FFFFFFF;
int ri = (raw >> 30) & 1;
int hc = (raw >> 31) & 1;
uint8_t *combo = all_combos + h * MAX_T;
fprintf(ff, "indices=");
printf("  indices=");
for (int j = 0; j < t_sel; j++) {
fprintf(ff, "%s%d", j?",":"", combo[j]);
printf("%s%d", j?",":"", combo[j]);
}


fprintf(ff, "\nhash_choice=%d\nrecid=%d\ncombo_idx=%d\n", hc, ri, combo_idx);
printf(" hc=%d recid=%d combo_idx=%d\n", hc, ri, combo_idx);
}
fclose(ff);






if (summary_f) {
time_t now_epoch = time(NULL);
for (int h = 0; h < nh; h++) {
uint32_t raw = hits[h];
int combo_idx = raw & 0x3FFFFFFF;
int ri = (raw >> 30) & 1;
int hc = (raw >> 31) & 1;
uint8_t *combo = all_combos + h * MAX_T;
fprintf(summary_f, "HIT %ld combo=", (long)now_epoch);
for (int j = 0; j < t_sel; j++)
fprintf(summary_f, "%s%d", j?",":"", combo[j]);
fprintf(summary_f, " hash_choice=%d recid=%d", hc, ri);
fprintf(summary_f, " combo_idx=%d calibrate=%d\n",
combo_idx, calibrate);
hit_counter++;
g_hit_counter = hit_counter;
}
fflush(summary_f);
fsync(fileno(summary_f));
}
}
}


if ((total_searched % 1000000) < (uint64_t)BATCH) {
for (int g = 0; g < num_gpus; g++) {
if (g == gpu_index) continue;
char check[256];
snprintf(check, sizeof(check), "results/digest_hit_%d.txt", g);
FILE *cf = fopen(check, "r");
if (cf) { fclose(cf); printf("  GPU %d found hit\n", g); found = 1; break; }
}
}




{
struct timespec t_now;
clock_gettime(CLOCK_MONOTONIC, &t_now);
double secs_since_report = (t_now.tv_sec - t_last_report.tv_sec)
+ (t_now.tv_nsec - t_last_report.tv_nsec) / 1e9;
if (secs_since_report >= 60.0) {
double elapsed_total = (t_now.tv_sec - t0.tv_sec)
+ (t_now.tv_nsec - t0.tv_nsec) / 1e9;
double rate = total_searched / elapsed_total;
double pct = (my_slice_total > 0) ? 100.0 * total_searched / (double)my_slice_total : 0.0;
double remaining_sec = (rate > 0 && my_slice_total > total_searched)
? (double)(my_slice_total - total_searched) / rate : 0.0;
int eta_h = (int)(remaining_sec / 3600);
int eta_m = (int)((remaining_sec - eta_h*3600) / 60);
printf("  [GPU %d] first=%d/%d  %.4f%% (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs  ETA=%dh%02dm\n",
gpu_index, first, n_pool - t_sel,
pct,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(my_slice_total/1000000),
rate/1e6, elapsed_total, eta_h, eta_m);
fflush(stdout);
if (summary_f) {
time_t now_epoch = time(NULL);
fprintf(summary_f,
"PROGRESS %ld first=%d attempts=%llu pct=%.4f rate_M_per_s=%.1f elapsed_s=%.0f eta=%dh%02dm hits_so_far=%llu\n",
(long)now_epoch, first,
(unsigned long long)total_searched, pct,
rate/1e6, elapsed_total, eta_h, eta_m,
(unsigned long long)hit_counter);
fflush(summary_f);


static double last_fsync = 0;
if (elapsed_total - last_fsync > 300) {
fsync(fileno(summary_f));
last_fsync = elapsed_total;
}
}
t_last_report = t_now;
}
}
}


clock_gettime(CLOCK_MONOTONIC, &t1);
double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
{
double rate = total_searched / elapsed;
double pct = (my_slice_total > 0) ? 100.0 * total_searched / (double)my_slice_total : 0.0;
double remaining_sec = (rate > 0 && my_slice_total > total_searched)
? (double)(my_slice_total - total_searched) / rate : 0.0;
int eta_h = (int)(remaining_sec / 3600);
int eta_m = (int)((remaining_sec - eta_h*3600) / 60);
printf("  [GPU %d] first=%d/%d DONE  %.2f%% (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs  ETA=%dh%02dm\n",
gpu_index, first, n_pool - t_sel,
pct,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(my_slice_total/1000000),
rate/1e6, elapsed, eta_h, eta_m);
}
}

clock_gettime(CLOCK_MONOTONIC, &t1);
double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
printf("\n  [GPU %d] Done: %lluM (of %lluM slice) in %.0fs (%.1fM/s), found=%d\n",
gpu_index,
(unsigned long long)(total_searched/1000000),
(unsigned long long)(my_slice_total/1000000),
elapsed, total_searched/elapsed/1e6, found);




if (summary_f) {
time_t now_epoch = time(NULL);
const char *status = found ? "FOUND" : "EXHAUSTED";
fprintf(summary_f,
"STATUS=%s %ld total_attempts=%llu slice_total=%llu elapsed_s=%.0f hits=%llu\n",
status, (long)now_epoch,
(unsigned long long)total_searched,
(unsigned long long)my_slice_total,
elapsed, (unsigned long long)hit_counter);
fflush(summary_f);
fsync(fileno(summary_f));
fclose(summary_f);
}

free(h_combos);
return 0;
#endif
}
