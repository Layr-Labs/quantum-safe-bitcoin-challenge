/* Host-only, default-off diagnostic for the two-slot producer pipeline.
 * No kernel/argument/enumeration changes. Events are harvested ONLY after the
 * existing slot completion fence, before the slot's timing records are reused.
 * The common origin is a one-time stream dependency, not a per-batch fence. */
#ifndef QSB_PRODUCER_CENSUS_H
#define QSB_PRODUCER_CENSUS_H
struct QsbProducerCensus {
cudaEvent_t origin = nullptr, ev[2][4] = {};
FILE *fp = nullptr;
uint64_t batch[2] = {};
int epochs[2] = {}, host[2] = {};
double check_ms[2] = {}, acquire_ms[2] = {};
static double now_ms() {
struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
cudaError_t init(int gpu, cudaStream_t *st) {
char name[128]; snprintf(name, sizeof name, "results/producer_census_gpu%d.csv", gpu);
fp = fopen(name, "w");
if (!fp) return cudaErrorUnknown;
fprintf(fp, "batch,slot,epochs,host,check_ms,acquire_ms,producer_begin_ms,first_begin_ms,digest_begin_ms,digest_end_ms\n");
cudaError_t e = cudaEventCreate(&origin);
for (int s=0; s<2 && e==cudaSuccess; ++s)
for (int p=0; p<4 && e==cudaSuccess; ++p) e=cudaEventCreate(&ev[s][p]);
if (e==cudaSuccess) e=cudaEventRecord(origin, st[0]);
for(int s=0;s<2 && e==cudaSuccess;++s) e=cudaStreamWaitEvent(st[s],origin,0);
return e;
}
void start(int s, uint64_t b, int n) {
batch[s]=b; epochs[s]=n; host[s]=0; check_ms[s]=acquire_ms[s]=0;
}
cudaError_t mark(int s, int p, cudaStream_t st) {
return cudaEventRecord(ev[s][p], st);
}
cudaError_t collect(int s) {
float t[4]; cudaError_t e=cudaSuccess;
for (int p=0; p<4 && e==cudaSuccess; ++p) e=cudaEventElapsedTime(&t[p],origin,ev[s][p]);
if(e!=cudaSuccess) return e;
if(fprintf(fp,"%llu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
(unsigned long long)batch[s],s,epochs[s],host[s],check_ms[s],acquire_ms[s],t[0],t[1],t[2],t[3])<0) return cudaErrorUnknown;
return cudaSuccess;
}
cudaError_t finish() {
cudaError_t e=cudaSuccess;
if(fp) { if(fclose(fp)!=0) e=cudaErrorUnknown; fp=nullptr; }
for(int s=0;s<2;++s) for(int p=0;p<4;++p) if(ev[s][p]) {
cudaError_t x=cudaEventDestroy(ev[s][p]); if(e==cudaSuccess)e=x; ev[s][p]=nullptr;
}
if(origin) { cudaError_t x=cudaEventDestroy(origin); if(e==cudaSuccess)e=x; origin=nullptr; }
return e;
}
~QsbProducerCensus() { (void)finish(); }
};
#endif
