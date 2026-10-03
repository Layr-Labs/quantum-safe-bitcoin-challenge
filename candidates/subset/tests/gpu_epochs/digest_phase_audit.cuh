/* Diagnostic-only sampled digest phase timestamps; GPL-3 inherited.
 * Disabled by default. clock64 records issue boundaries, NOT isolated service
 * time: latency/warp scheduling, barriers, queue overlap and instrumentation
 * perturbation are included. No throughput qualification from this image.
 */
#pragma once
#ifndef QSB_DIGEST_PHASE_AUDIT
#define QSB_DIGEST_PHASE_AUDIT 0
#endif
#if QSB_DIGEST_PHASE_AUDIT
#if QSB_CARRIER
#error "Digest phase audit must bypass the embedded production carrier"
#endif
#define QSB_DP_ROWS 4096
#define QSB_DP_PERIOD 8192
__device__ uint64_t *qsb_dp_rows[2];
__device__ const uint32_t *qsb_dp_first[2];
__device__ __forceinline__ void qsb_dp_stamp(const uint32_t *first, int phase) {
if ((blockIdx.x % QSB_DP_PERIOD)==0 && (threadIdx.x % 32)==0) {
unsigned row=(blockIdx.x / QSB_DP_PERIOD)*4+threadIdx.x/32;
if(row<QSB_DP_ROWS) {
uint64_t now;asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
int slot=first==qsb_dp_first[1];
qsb_dp_rows[slot][row*7+phase]=now;
}
}
}
#define QSB_DP_STAMP(p) qsb_dp_stamp(d_first,p)
#else
#define QSB_DP_STAMP(p) ((void)0)
#endif

#if QSB_DIGEST_PHASE_AUDIT
struct QsbDigestPhaseAudit {
uint64_t *d[2]={nullptr,nullptr}, *h=nullptr;FILE *fp=nullptr;
static constexpr size_t bytes=QSB_DP_ROWS*7*sizeof(uint64_t);
cudaError_t init(uint32_t **first) {
fp=fopen("results/digest-phases.csv","w");if(!fp)return cudaErrorUnknown;
fprintf(fp,"batch,slot,epochs,sample,entry,sha,front_a,front_b,cross,inverse,tail\n");
h=(uint64_t*)malloc(bytes);if(!h)return cudaErrorMemoryAllocation;
cudaError_t e=cudaSuccess;
for(int s=0;s<2 && e==cudaSuccess;s++)e=cudaMalloc(&d[s],bytes);
if(e==cudaSuccess)e=cudaMemcpyToSymbol(qsb_dp_rows,d,sizeof(d));
if(e==cudaSuccess)e=cudaMemcpyToSymbol(qsb_dp_first,first,2*sizeof(*first));
return e;
}
cudaError_t clear(int slot,cudaStream_t st) {return cudaMemsetAsync(d[slot],0,bytes,st);}
cudaError_t collect(int slot,uint64_t batch,int epochs) {
cudaError_t e=cudaMemcpy(h,d[slot],bytes,cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return e;
for(int r=0;r<QSB_DP_ROWS;r++)if(h[r*7]) {
fprintf(fp,"%llu,%d,%d,%d",(unsigned long long)batch,slot,epochs,r);
for(int k=0;k<7;k++)fprintf(fp,",%llu",(unsigned long long)h[r*7+k]);
fputc('\n',fp);
}
return fflush(fp)==0?cudaSuccess:cudaErrorUnknown;
}
~QsbDigestPhaseAudit() {if(fp)fclose(fp);free(h);for(int s=0;s<2;s++)if(d[s])cudaFree(d[s]);}
};
#endif
