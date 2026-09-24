// SPDX-License-Identifier: GPL-3.0-only
// Optional native module transport. Device arithmetic remains in pinning.cu.
#pragma once
#include <cuda.h>
#include <dlfcn.h>
#include <unistd.h>
#include <openssl/sha.h>
#include <string>
#include <vector>
#include "NativeManifest.h"

#ifndef QSB_NATIVE_MODULE
#define QSB_NATIVE_MODULE 1
#endif

__device__ __constant__ unsigned char qsb_native_source_stamp[32] = QSB_NATIVE_BUILD_BYTES;

namespace qsb_native {
enum Kernel { Prepare, Finish, RootPrepare, RootInvert, RootFinish, BuildTable, OffsetY, KernelCount };
static const char *const kernel_names[KernelCount] = {
    "_Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre",
    "_Z23kernel_pinning_pipelineILb1ELi2EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_12qsb_tail_pre",
    "_Z22qsb_root_group_preparePKmiPmS1_", "_Z22qsb_invert_super_rootsPmi",
    "_Z21qsb_root_group_finishPmiPKmS1_", "_Z19kernel_build_gtablePKmS0_Ph", "_Z18qsb_table_offset_yPh"
};
struct Global { const char *name; size_t size; CUdeviceptr address; };
static Global globals[] = {
    {"pin_u2rx_words",32,0}, {"pin_u2ry_words",32,0}, {"pin_iso_invu_words",32,0},
    {"pin_iso_u2ry_words",32,0}, {"pin_iso_xneg",4,0}, {"pin_recovery_c",32,0},
    {"pin_u2rk_words",32,0}, {"pin_one_mul",4,0}, {"pin_tail_words",12,0},
    {"pin_tail_tab",4096,0}
};
static void *library = nullptr;
static CUmodule module = nullptr;
static CUfunction functions[KernelCount] = {};
static bool active = false;
static double startup_begin = 0;
static decltype(&cuModuleLoadData) module_load = nullptr;
static decltype(&cuModuleUnload) module_unload = nullptr;
static decltype(&cuModuleGetFunction) get_function = nullptr;
static decltype(&cuModuleGetGlobal) get_global = nullptr;
static decltype(&cuMemcpyHtoD) copy_htod = nullptr;
static decltype(&cuMemcpyDtoH) copy_dtoh = nullptr;
static decltype(&cuLaunchKernel) launch_kernel = nullptr;
static decltype(&cuModuleGetLoadingMode) get_loading_mode = nullptr;

static double now() {
    timespec value; clock_gettime(CLOCK_MONOTONIC,&value);
    return (double)value.tv_sec + (double)value.tv_nsec/1e9;
}
static bool configuration_matches() {
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) || defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM) || defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
    return false; // This module transport uses legacy-default driver entries.
#else
    return QSB_NATIVE_CONFIG_MATCH;
#endif
}
static void begin() {
    startup_begin=now();
#if QSB_NATIVE_MODULE
    // Registration of the unused runtime fatbin must not eagerly compile it.
    // There are no managed globals or device function-pointer globals here.
    if(configuration_matches()) setenv("CUDA_MODULE_LOADING","LAZY",1);
#endif
}
static std::string hex_digest(const unsigned char *data,size_t bytes) {
    unsigned char hash[32]; SHA256(data,bytes,hash);
    char result[65]; for(int i=0;i<32;i++) snprintf(result+2*i,3,"%02x",hash[i]);
    return std::string(result,64);
}
static bool read_file(const std::string &path,std::vector<unsigned char> &bytes) {
    FILE *f=fopen(path.c_str(),"rb"); if(!f)return false;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);return false;}
    long size=ftell(f); if(size<0 || size>16*1024*1024){fclose(f);return false;}
    rewind(f); bytes.resize((size_t)size);
    bool ok=fread(bytes.data(),1,bytes.size(),f)==bytes.size();
    fclose(f); return ok;
}
static void fallback(const char *reason) {
    if(module && module_unload) module_unload(module);
    module=nullptr; active=false;
    if(library)dlclose(library);
    library=nullptr;
    fprintf(stderr,"Native module: runtime fallback (%s)\n",reason);
}
static void initialize(int major,int minor,const char *argv0) {
#if !QSB_NATIVE_MODULE
    (void)major;(void)minor;(void)argv0;
    fallback("disabled by QSB_NATIVE_MODULE=0"); return;
#else
    if(!configuration_matches()){fallback("compile configuration differs from native manifest");return;}
    if(major!=8 || minor!=9){fallback("device is not sm_89");return;}
    double before=now();
    char executable[4096]; ssize_t length=readlink("/proc/self/exe",executable,sizeof(executable)-1);
    std::string path;
    if(length>0){executable[length]=0;path=executable;}else path=argv0;
    size_t slash=path.find_last_of('/');
    if(slash==std::string::npos){fallback("cannot locate executable directory");return;}
    std::string directory=path.substr(0,slash+1);
    std::vector<unsigned char> bytes;
    for(const auto &source : native_manifest_sources) {
        if(!read_file(directory+source.name,bytes) || hex_digest(bytes.data(),bytes.size())!=source.sha256){
            fallback("source closure hash mismatch or missing source");return;
        }
    }
    if(!read_file(directory+"pinning_sm89.cubin",bytes) ||
       hex_digest(bytes.data(),bytes.size())!=QSB_NATIVE_CUBIN_SHA256){
        fallback("native cubin missing or hash mismatch");return;
    }
    // ELF64 little-endian NVIDIA cubin, native architecture sm_89.
    if(bytes.size()<64 || memcmp(bytes.data(),"\177ELF",4)!=0 || bytes[4]!=2 || bytes[5]!=1 ||
       bytes[18]!=190 || bytes[19]!=0 || bytes[48]!=89){fallback("invalid native cubin architecture");return;}
    double checked=now();
    library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!library){fallback("libcuda.so.1 unavailable");return;}
#define QSB_NATIVE_RESOLVE(variable,name) \
    variable=reinterpret_cast<decltype(variable)>(dlsym(library,name)); \
    if(!variable){fallback("missing driver entry " name);return;}
    QSB_NATIVE_RESOLVE(module_load,"cuModuleLoadData")
    QSB_NATIVE_RESOLVE(module_unload,"cuModuleUnload")
    QSB_NATIVE_RESOLVE(get_function,"cuModuleGetFunction")
    QSB_NATIVE_RESOLVE(get_global,"cuModuleGetGlobal_v2")
    QSB_NATIVE_RESOLVE(copy_htod,"cuMemcpyHtoD_v2")
    QSB_NATIVE_RESOLVE(copy_dtoh,"cuMemcpyDtoH_v2")
    QSB_NATIVE_RESOLVE(launch_kernel,"cuLaunchKernel")
    QSB_NATIVE_RESOLVE(get_loading_mode,"cuModuleGetLoadingMode")
#undef QSB_NATIVE_RESOLVE
    CUmoduleLoadingMode loading_mode=CU_MODULE_EAGER_LOADING;
    if(get_loading_mode(&loading_mode)!=CUDA_SUCCESS || loading_mode!=CU_MODULE_LAZY_LOADING){
        fallback("driver did not select LAZY module loading");return;
    }
    if(module_load(&module,bytes.data())!=CUDA_SUCCESS){fallback("native module load failed");return;}
    for(int i=0;i<KernelCount;i++) {
        if(get_function(&functions[i],module,kernel_names[i])!=CUDA_SUCCESS){fallback("native kernel symbol missing");return;}
    }
    for(auto &global:globals) {
        size_t size=0;
        if(get_global(&global.address,&size,module,global.name)!=CUDA_SUCCESS || size!=global.size){
            fallback("native constant symbol or size mismatch");return;
        }
    }
    CUdeviceptr stamp=0; size_t stamp_size=0;
    const unsigned char expected[32]=QSB_NATIVE_BUILD_BYTES;
    unsigned char observed[32]={};
    if(get_global(&stamp,&stamp_size,module,"qsb_native_source_stamp")!=CUDA_SUCCESS || stamp_size!=32 ||
       copy_dtoh(observed,stamp,32)!=CUDA_SUCCESS || memcmp(expected,observed,32)!=0){
        fallback("native source/configuration stamp mismatch");return;
    }
    active=true;
    fprintf(stderr,"Native module: active sm_89; loading mode=LAZY; source validation %.6f s; driver load %.6f s\n",checked-before,now()-checked);
#endif
}
static cudaError_t copy_constant(const char *name,const void *source,size_t bytes) {
    for(const auto &global:globals) if(strcmp(name,global.name)==0 && bytes==global.size) {
        // Once selected, native failures are fatal to the caller; never mix modules.
        if(copy_htod(global.address,source,bytes)!=CUDA_SUCCESS)return cudaErrorUnknown;
        std::vector<unsigned char> check(bytes);
        if(copy_dtoh(check.data(),global.address,bytes)!=CUDA_SUCCESS || memcmp(check.data(),source,bytes)!=0)return cudaErrorUnknown;
        return cudaSuccess;
    }
    return cudaErrorInvalidSymbol;
}
static cudaError_t launch(Kernel kernel,unsigned blocks,unsigned threads,cudaStream_t stream,void **arguments) {
    return launch_kernel(functions[kernel],blocks,1,1,threads,1,1,0,reinterpret_cast<CUstream>(stream),arguments,nullptr)==CUDA_SUCCESS
        ? cudaSuccess : cudaErrorUnknown;
}
template<typename... Arguments>
static cudaError_t launch_args(Kernel kernel,unsigned blocks,unsigned threads,cudaStream_t stream,const Arguments &...values) {
    const unsigned expected[KernelCount]={23,23,4,2,4,3,1};
    if(sizeof...(Arguments)!=expected[kernel])return cudaErrorInvalidValue;
    // Driver metadata supplies each parameter's size/alignment, including the
    // 76-byte qsb_tail_pre value. Each entry points at the original host value.
    void *arguments[]={const_cast<void*>(static_cast<const void*>(&values))...};
    return launch(kernel,blocks,threads,stream,arguments);
}
static void report_startup() {
    fprintf(stderr,"Initialization before search: %.6f seconds; execution backend=%s\n",now()-startup_begin,active?"native_sm89":"runtime");
}
} // namespace qsb_native

// The inactive branch is never evaluated after the native module is selected.
#define QSB_COPY_CONSTANT(symbol,source,bytes) \
    (qsb_native::active ? qsb_native::copy_constant(#symbol,source,bytes) : cudaMemcpyToSymbol(symbol,source,bytes))
