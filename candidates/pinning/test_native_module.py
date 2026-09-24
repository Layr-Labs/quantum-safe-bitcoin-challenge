#!/usr/bin/env python3
"""Exercise the actual native loader with a CPU mock CUDA driver, never a GPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

CUDA = r"""
#pragma once
#include <cstddef>
#include <cstdint>
#define __device__
#define __constant__
using CUresult=int;
using CUmodule=void*;
using CUfunction=void*;
using CUstream=void*;
using CUdeviceptr=uintptr_t;
using cudaStream_t=void*;
using cudaError_t=int;
using CUmoduleLoadingMode=int;
enum { CUDA_SUCCESS=0, cudaSuccess=0, cudaErrorUnknown=999, cudaErrorInvalidSymbol=13,
       cudaErrorInvalidValue=1, CU_MODULE_EAGER_LOADING=1, CU_MODULE_LAZY_LOADING=2 };
CUresult cuModuleLoadData(CUmodule*,const void*);
CUresult cuModuleUnload(CUmodule);
CUresult cuModuleGetFunction(CUfunction*,CUmodule,const char*);
CUresult cuModuleGetGlobal(CUdeviceptr*,size_t*,CUmodule,const char*);
CUresult cuMemcpyHtoD(CUdeviceptr,const void*,size_t);
CUresult cuMemcpyDtoH(void*,CUdeviceptr,size_t);
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode*);
CUresult cuLaunchKernel(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
                        unsigned,CUstream,void**,void**);
"""

CPP = r"""
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>
#include <openssl/sha.h>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "cuda.h"
static bool config_match=true;
static std::string scenario;
static int opened=0,closed=0,loaded=0,unloaded=0,resolved=0,functions_seen=0;
static int globals_seen=0,reads=0,writes=0,launches=0;
static CUmodule mock_module=reinterpret_cast<CUmodule>(0x2000);
static CUstream expected_stream=reinterpret_cast<CUstream>(0x76543210);
static void** expected_args=nullptr;
static int expected_kernel=0;
static std::vector<void*> packed_addresses;
static std::map<std::string,std::vector<unsigned char>> memory;
static void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
static void* mock_dlopen(const char*,int);
static void* mock_dlsym(void*,const char*);
static int mock_dlclose(void*);
#define dlopen mock_dlopen
#define dlsym mock_dlsym
#define dlclose mock_dlclose
#include "NativeModule.h"
#undef dlopen
#undef dlsym
#undef dlclose

static void* mock_dlopen(const char* name,int flags) {
    require(std::string(name)=="libcuda.so.1","wrong driver library");
    require(flags==(RTLD_NOW|RTLD_LOCAL),"wrong loader flags");
    if(scenario=="library")return nullptr;
    ++opened;return reinterpret_cast<void*>(0x1000);
}
static int mock_dlclose(void* handle) {
    require(handle==reinterpret_cast<void*>(0x1000),"wrong library close");
    ++closed;return 0;
}
static void* mock_dlsym(void*,const char* name) {
    ++resolved;
    if(scenario==std::string("entry:")+name)return nullptr;
#define ENTRY(symbol,func) if(std::string(name)==symbol)return reinterpret_cast<void*>(&func);
    ENTRY("cuModuleLoadData",cuModuleLoadData)
    ENTRY("cuModuleUnload",cuModuleUnload)
    ENTRY("cuModuleGetFunction",cuModuleGetFunction)
    ENTRY("cuModuleGetGlobal_v2",cuModuleGetGlobal)
    ENTRY("cuMemcpyHtoD_v2",cuMemcpyHtoD)
    ENTRY("cuMemcpyDtoH_v2",cuMemcpyDtoH)
    ENTRY("cuLaunchKernel",cuLaunchKernel)
    ENTRY("cuModuleGetLoadingMode",cuModuleGetLoadingMode)
#undef ENTRY
    throw std::runtime_error("unexpected driver entry");
}
CUresult cuModuleLoadData(CUmodule* output,const void* data) {
    require(std::memcmp(data,"\177ELF",4)==0,"driver received non-ELF module");
    if(scenario=="load")return 17;
    ++loaded;*output=mock_module;return CUDA_SUCCESS;
}
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) {
    *mode=scenario=="eager"?CU_MODULE_EAGER_LOADING:CU_MODULE_LAZY_LOADING;
    return scenario=="loading-mode"?23:CUDA_SUCCESS;
}
CUresult cuModuleUnload(CUmodule module) {
    require(module==mock_module,"wrong module unload");
    ++unloaded;return CUDA_SUCCESS;
}
CUresult cuModuleGetFunction(CUfunction* output,CUmodule module,const char* name) {
    require(module==mock_module,"wrong function module");
    ++functions_seen;
    require(std::string(name)==qsb_native::kernel_names[functions_seen-1],"kernel resolution order");
    if(scenario=="function:"+std::to_string(functions_seen-1))return 18;
    *output=reinterpret_cast<CUfunction>(static_cast<uintptr_t>(0x3000+functions_seen-1));
    return CUDA_SUCCESS;
}
CUresult cuModuleGetGlobal(CUdeviceptr* output,size_t* bytes,CUmodule module,const char* name) {
    require(module==mock_module,"wrong global module");
    ++globals_seen;
    auto found=memory.find(name);
    require(found!=memory.end(),"unexpected global name");
    if(scenario==std::string("global:")+name)return 19;
    *output=reinterpret_cast<CUdeviceptr>(found->second.data());
    *bytes=found->second.size()+(scenario==std::string("size:")+name);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyHtoD(CUdeviceptr destination,const void* source,size_t bytes) {
    ++writes;
    if(scenario=="copy-write")return 20;
    std::memcpy(reinterpret_cast<void*>(destination),source,bytes);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyDtoH(void* destination,CUdeviceptr source,size_t bytes) {
    ++reads;
    if(scenario=="stamp-read" || (scenario=="copy-read" && reads>1))return 21;
    std::memcpy(destination,reinterpret_cast<void*>(source),bytes);
    if(scenario=="copy-corrupt" && reads>1)static_cast<unsigned char*>(destination)[0]^=1;
    return CUDA_SUCCESS;
}
CUresult cuLaunchKernel(CUfunction function,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
    ++launches;
    require(function==reinterpret_cast<CUfunction>(static_cast<uintptr_t>(0x3000+expected_kernel)),"wrong native kernel");
    require(gx==17 && gy==1 && gz==1 && bx==128 && by==1 && bz==1 && shared==0,"launch geometry changed");
    require(stream==expected_stream,"runtime/driver stream identity changed");
    require(!extra,"unexpected packed-buffer launch");
    if(!packed_addresses.empty()) {
        for(size_t i=0;i<packed_addresses.size();i++)
            require(args[i]==packed_addresses[i],"typed parameter address/order changed");
        return scenario=="launch"?22:CUDA_SUCCESS;
    }
    require(args==expected_args,"argument array was replaced");
    require(*static_cast<uint64_t*>(args[0])==0xfedcba9876543210ULL,"pointer-sized argument changed");
    require(*static_cast<int*>(args[1])==-2147483600,"signed argument changed");
    const unsigned char* tail=static_cast<const unsigned char*>(args[2]);
    for(int i=0;i<76;i++)require(tail[i]==static_cast<unsigned char>(3*i+1),"by-value tail argument changed");
    return scenario=="launch"?22:CUDA_SUCCESS;
}

int main(int argc,char** argv) {
    try {
        require(argc==2,"one scenario required");scenario=argv[1];
        const std::pair<const char*,size_t> expected_globals[]={
            {"pin_u2rx_words",32},{"pin_u2ry_words",32},{"pin_iso_invu_words",32},
            {"pin_iso_u2ry_words",32},{"pin_iso_xneg",4},{"pin_recovery_c",32},
            {"pin_u2rk_words",32},{"pin_one_mul",4},{"pin_tail_words",12},{"pin_tail_tab",4096}
        };
        require(qsb_native::KernelCount==7,"missing production kernel");
        require(sizeof(qsb_native::globals)/sizeof(qsb_native::globals[0])==10,"missing production constant");
        for(int i=0;i<10;i++) {
            require(std::string(qsb_native::globals[i].name)==expected_globals[i].first &&
                    qsb_native::globals[i].size==expected_globals[i].second,"constant layout differs");
            memory[expected_globals[i].first].resize(expected_globals[i].second);
        }
        const unsigned char stamp[32]=QSB_NATIVE_BUILD_BYTES;
        memory["qsb_native_source_stamp"]=std::vector<unsigned char>(stamp,stamp+32);
        if(scenario=="stamp")memory["qsb_native_source_stamp"][0]^=1;
        if(scenario=="config")config_match=false;
        unsetenv("CUDA_MODULE_LOADING");
        qsb_native::begin();
        require((getenv("CUDA_MODULE_LOADING")!=nullptr)==(qsb_native::configuration_matches() && QSB_NATIVE_MODULE),"lazy-loading config gate differs");
        qsb_native::initialize(scenario=="major"?9:8,scenario=="minor"?0:9,argv[0]);
        const bool success=scenario=="success" || scenario.find("copy-")==0 || scenario=="launch";
        require(qsb_native::active==success,"unexpected selected backend");
        if(!success) {
            require(writes==0 && launches==0,"fallback occurred after native work");
            require(unloaded==loaded && closed==opened,"partial initialization leaked resources");
            if(scenario=="config" || scenario=="major" || scenario=="minor" || scenario=="disabled" || scenario=="per-thread" ||
               scenario=="source" || scenario=="cubin" || scenario=="architecture")
                require(opened==0 && loaded==0,"unsafe input reached driver");
        } else {
            require(functions_seen==7 && globals_seen==11 && reads==1,"incomplete module validation");
            std::vector<unsigned char> value(4096);
            for(size_t i=0;i<value.size();i++)value[i]=static_cast<unsigned char>(7*i+3);
            int before=writes;
            require(qsb_native::copy_constant("not_a_constant",value.data(),4)==cudaErrorInvalidSymbol,"unknown constant accepted");
            require(qsb_native::copy_constant("pin_iso_xneg",value.data(),8)==cudaErrorInvalidSymbol,"wrong constant size accepted");
            require(writes==before,"invalid constant was copied");
            for(const auto& item:expected_globals) {
                auto result=qsb_native::copy_constant(item.first,value.data(),item.second);
                require(result==(scenario.find("copy-")==0?cudaErrorUnknown:cudaSuccess),"constant failure swallowed");
                require(qsb_native::active && !unloaded && !closed,"mid-run constant failure switched backend");
                if(scenario.find("copy-")==0)break;
                require(std::memcmp(memory[item.first].data(),value.data(),item.second)==0,"constant bytes changed");
            }
            uint64_t pointer=0xfedcba9876543210ULL;int signed_value=-2147483600;
            unsigned char tail[76];for(int i=0;i<76;i++)tail[i]=static_cast<unsigned char>(3*i+1);
            void* args[]={&pointer,&signed_value,tail};expected_args=args;
            for(int i=0;i<7;i++) {
                expected_kernel=i;
                auto result=qsb_native::launch(static_cast<qsb_native::Kernel>(i),17,128,
                                               reinterpret_cast<cudaStream_t>(expected_stream),args);
                require(result==(scenario=="launch"?cudaErrorUnknown:cudaSuccess),"launch failure swallowed");
                require(qsb_native::active && !unloaded && !closed,"mid-run launch failure switched backend");
            }
            // Execute the actual variadic packer with the complete heterogeneous
            // pipeline ABI, including the 76-byte by-value SHA-tail object.
            const uint32_t *mid=nullptr;const uint8_t *suffix=nullptr;
            int suffix_len=75,seq_offset=31,lt_offset=67,total_length=9995;
            uint32_t sequence=0x80000012,start_lt=500000000;
            const uint64_t *nri=nullptr,*x=nullptr,*y=nullptr,*nx=nullptr,*ny=nullptr;
            uint8_t *table=nullptr;uint32_t *count=nullptr,*hits=nullptr;
            int batch=8388608,easy=0,single=1;
            struct alignas(16) Vector {uint64_t x,y;};Vector *saved=nullptr;
            uint64_t *roots=nullptr,*tree=nullptr;
            struct Tail {uint32_t v[19];} tp={};static_assert(sizeof(tp)==76,"tail ABI");
            packed_addresses={&mid,&suffix,&suffix_len,&seq_offset,&lt_offset,&total_length,
                &sequence,&start_lt,&nri,&x,&y,&nx,&ny,&table,&count,&hits,&batch,&easy,&single,
                &saved,&roots,&tree,&tp};
            for(int kernel=0;kernel<2;kernel++) {
                expected_kernel=kernel;
                auto result=qsb_native::launch_args(static_cast<qsb_native::Kernel>(kernel),17,128,
                    reinterpret_cast<cudaStream_t>(expected_stream),mid,suffix,suffix_len,seq_offset,
                    lt_offset,total_length,sequence,start_lt,nri,x,y,nx,ny,table,count,hits,
                    batch,easy,single,saved,roots,tree,tp);
                require(result==(scenario=="launch"?cudaErrorUnknown:cudaSuccess),"typed pipeline launch failed");
            }
            int prior=launches;
            require(qsb_native::launch_args(qsb_native::OffsetY,17,128,expected_stream,table,batch)==
                    cudaErrorInvalidValue,"wrong argument count accepted");
            require(launches==prior,"wrong argument count reached driver");
            qsb_native::report_startup();
            qsb_native::fallback("test cleanup");
            require(!qsb_native::active && unloaded==1 && closed==1,"cleanup failed");
        }
        std::cout<<"PASS "<<scenario<<"\n";return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}
"""


class NativeModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qsb-native-mock-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.header = (HERE / "NativeModule.h").read_text()
        cls.build("normal")
        cls.build("disabled", enabled=0)
        cls.build("architecture", architecture=88)
        for macro in ("CUDA_API_PER_THREAD_DEFAULT_STREAM", "__CUDART_API_PER_THREAD_DEFAULT_STREAM", "__CUDA_API_PER_THREAD_DEFAULT_STREAM"):
            cls.build(macro, stream_macro=macro)

    @classmethod
    def build(cls, name, enabled=1, architecture=89, stream_macro=None):
        directory = cls.root / name
        directory.mkdir()
        (directory / "NativeModule.h").write_text(cls.header)
        (directory / "cuda.h").write_text(CUDA)
        (directory / "test.cpp").write_text(CPP)
        source = b"mock source closure fixture, never GPU source\n"
        cubin = bytearray(64)
        cubin[:6] = b"\x7fELF\x02\x01"
        cubin[18] = 190
        cubin[48] = architecture
        (directory / "fixture.cu").write_bytes(source)
        (directory / "pinning_sm89.cubin").write_bytes(cubin)
        manifest = """#pragma once
#define QSB_NATIVE_CONFIG_MATCH config_match
#define QSB_NATIVE_BUILD_BYTES {%s}
#define QSB_NATIVE_CUBIN_SHA256 "%s"
struct ManifestSource {const char *name,*sha256;};
static const ManifestSource native_manifest_sources[]={{"fixture.cu","%s"}};
""" % (",".join(str(i) for i in range(32)), hashlib.sha256(cubin).hexdigest(), hashlib.sha256(source).hexdigest())
        (directory / "NativeManifest.h").write_text(manifest)
        command = [os.environ.get("CXX", "g++"), "-std=c++11", "-O1", "-Wall", "-Wextra",
                   "-fsanitize=undefined", "-fno-sanitize-recover=undefined", f"-DQSB_NATIVE_MODULE={enabled}",
                   "-I", str(directory), str(directory / "test.cpp"), "-lcrypto", "-ldl", "-o", str(directory / "test")]
        if stream_macro:
            command.insert(1, "-D" + stream_macro)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)

    def run_case(self, scenario, binary="normal"):
        result = subprocess.run([str(self.root / binary / "test"), scenario], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + scenario, result.stdout)

    def test_success_constant_bytes_launch_geometry_and_stream_identity(self):
        self.run_case("success")

    def test_unsupported_configuration_and_device_never_open_driver(self):
        for scenario in ("config", "major", "minor"):
            with self.subTest(scenario=scenario):
                self.run_case(scenario)
        self.run_case("disabled", "disabled")
        for macro in ("CUDA_API_PER_THREAD_DEFAULT_STREAM", "__CUDART_API_PER_THREAD_DEFAULT_STREAM", "__CUDA_API_PER_THREAD_DEFAULT_STREAM"):
            self.run_case("per-thread", macro)

    def test_source_cubin_hashes_and_architecture_fail_before_driver(self):
        directory = self.root / "normal"
        for name, scenario in (("fixture.cu", "source"), ("pinning_sm89.cubin", "cubin")):
            path = directory / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"tamper")
                self.run_case(scenario)
                path.unlink()
                self.run_case(scenario)
            finally:
                path.write_bytes(original)
        self.run_case("architecture", "architecture")

    def test_driver_resolution_load_and_every_kernel_failure_clean_up(self):
        names = ("cuModuleLoadData", "cuModuleUnload", "cuModuleGetFunction", "cuModuleGetGlobal_v2",
                 "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuLaunchKernel", "cuModuleGetLoadingMode")
        for scenario in ["library", "load", "eager", "loading-mode"] + ["entry:" + n for n in names] + [f"function:{i}" for i in range(7)]:
            with self.subTest(scenario=scenario):
                self.run_case(scenario)

    def test_every_constant_size_and_stamp_failure_clean_up(self):
        names = ("pin_u2rx_words", "pin_u2ry_words", "pin_iso_invu_words", "pin_iso_u2ry_words",
                 "pin_iso_xneg", "pin_recovery_c", "pin_u2rk_words", "pin_one_mul", "pin_tail_words",
                 "pin_tail_tab", "qsb_native_source_stamp")
        for scenario in [prefix + n for prefix in ("global:", "size:") for n in names] + ["stamp", "stamp-read"]:
            with self.subTest(scenario=scenario):
                self.run_case(scenario)

    def test_failures_after_selection_propagate_without_backend_switch(self):
        for scenario in ("copy-write", "copy-read", "copy-corrupt", "launch"):
            with self.subTest(scenario=scenario):
                self.run_case(scenario)

    def test_actual_production_routes_match_runtime_parameter_order(self):
        source = (HERE / "pinning.cu").read_text()
        routes = {
            "Prepare": r"kernel_pinning_pipeline<FAST_TAIL,0>",
            "Finish": r"kernel_pinning_pipeline<FAST_TAIL,2>",
            "RootPrepare": "qsb_root_group_prepare",
            "RootInvert": "qsb_invert_super_roots",
            "RootFinish": "qsb_root_group_finish",
            "BuildTable": "kernel_build_gtable",
            "OffsetY": "qsb_table_offset_y",
        }
        def split_arguments(text):
            result, start, depth = [], 0, 0
            for i, char in enumerate(text):
                depth += (char == "(") - (char == ")")
                if char == "," and depth == 0:
                    result.append(re.sub(r"\s+", "", text[start:i]))
                    start = i + 1
            result.append(re.sub(r"\s+", "", text[start:]))
            return result
        for kernel, runtime in routes.items():
            with self.subTest(kernel=kernel):
                native = re.findall(r"launch_args\(qsb_native::" + kernel + r",(.*?)\);", source, re.S)
                self.assertEqual(len(native), 1, "native kernel route must be unique")
                geometry, arguments = re.search(runtime + r"<<<(.*?)>>>\((.*?)\);", source, re.S).groups()
                native_args = split_arguments(native[0])
                runtime_geometry = split_arguments(geometry.replace("QSB_STREAM_ARG", ""))
                self.assertEqual(native_args[:2], runtime_geometry)
                self.assertEqual(native_args[3:], split_arguments(arguments))
                self.assertEqual(native_args[2], "nullptr" if kernel in ("BuildTable", "OffsetY") else "QSB_NATIVE_STREAM")
        self.assertNotIn("cudaMemcpyToSymbol(", source, "runtime constant upload bypasses native routing")
        uploaded = set(re.findall(r"QSB_COPY_CONSTANT\((pin_\w+),", source))
        expected = set(re.findall(r'\{"(pin_\w+)",\d+,0\}', self.header))
        self.assertEqual(uploaded, expected)
        pipeline = source[source.index("static void launch_pinning_pipeline("):source.index("/* ============================================================", source.index("static void launch_pinning_pipeline("))]
        positions = [pipeline.index(token) for token in (
            "qsb_native::Prepare", "flow->begin_roots(st)", "qsb_native::RootPrepare",
            "qsb_native::RootInvert", "qsb_native::RootFinish", "flow->end_roots(st)", "qsb_native::Finish")]
        self.assertEqual(positions, sorted(positions), "native route moved across an event handoff")
        geometry = re.search(r"const bool native_geometry\s*=([^;]+);", source).group(1)
        fast_tail = re.search(r"const bool fast_tail\s*=([^;]+);", source).group(1)
        self.assertEqual(re.sub(r"\s+", "", geometry), re.sub(r"\s+", "", fast_tail))
        self.assertIn("if(native_geometry) qsb_native::initialize", source)
        self.assertLess(source.index("if(native_geometry) qsb_native::initialize"), source.index("qsb_native::BuildTable"))
        symbols = re.findall(r'"(_Z[A-Za-z0-9_]+)"', self.header)
        decoded = subprocess.check_output(["c++filt"] + symbols, text=True).splitlines()
        expected_names = ("kernel_pinning_pipeline<true, 0>", "kernel_pinning_pipeline<true, 2>",
                          "qsb_root_group_prepare", "qsb_invert_super_roots", "qsb_root_group_finish",
                          "kernel_build_gtable", "qsb_table_offset_y")
        self.assertEqual(len(decoded), len(expected_names))
        for name, signature in zip(expected_names, decoded):
            self.assertRegex(signature, r"^(void )?" + re.escape(name) + r"\(")

    def test_actual_manifest_rejects_changed_math_and_layout_configuration(self):
        build = json.loads((HERE / "research/native_module_build.json").read_text())
        definitions = ["#define " + name + " " + value for name, value in build["configuration"].items() if value is not None]
        changes = {
            "QSB_ZEROS_N": "23", "QSB_MAC_HALF_SEED": "1", "QSB_GLV_COLD_SEED": "1",
            "QSB_S0_THREADS": "64", "QSB_YOFF": "0", "QSB_ISO_XR": "0", "QSB_TREE_OFFLOAD": "1",
            "QSB_TAIL_PRE": "0", "QSB_BIGTBL_HOST_EXACT": "1",
        }
        for name, value in [(None, None)] + list(changes.items()):
            with self.subTest(configuration=name or "baseline"):
                text = "\n".join(definitions) + "\n"
                if name:
                    text += f"#undef {name}\n#define {name} {value}\n"
                text += '#include "NativeManifest.h"\nint main(){return QSB_NATIVE_CONFIG_MATCH ? 0 : 23;}\n'
                path = self.root / "guard.cpp"
                path.write_text(text)
                binary = self.root / "guard"
                compile_result = subprocess.run([os.environ.get("CXX", "g++"), "-std=c++11", "-I", str(HERE), str(path), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                result = subprocess.run([str(binary)])
                self.assertEqual(result.returncode, 23 if name else 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
