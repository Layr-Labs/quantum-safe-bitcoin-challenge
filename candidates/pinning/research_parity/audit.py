#!/usr/bin/env python3
"""CPU audit of the narrowed parity window, including actual fallback PTX.

Preprocess both switch settings, translate the candidate's integer PTX into
unsigned host C++, and compare two million deterministic rows plus guard and
carry boundaries. This checks PTX arithmetic semantics, not CUDA codegen,
device execution, or GPU speed. Requires only Python and g++.
SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parent
HERE = LAB.parent
DEFINES = {
    "QSB_HOST_GATE": 1, "QSB_CARRY62": 1, "QSB_C31": 1, "QSB_SHORT_CARRY": 1, "QSB_FIELD_SC": 1,
    "QSB_PARITY_SUM": 1, "__CUDA_ARCH__": 520,
}


def preprocess(path, narrow):
    args = ["g++", "-E", "-P", "-x", "c++"]
    args += [f"-D{k}={v}" for k, v in DEFINES.items()]
    args += [f"-DQSB_PARITY_WINDOW_NARROW={narrow}", str(path)]
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE)


def function(text, name):
    start = text.rfind("__device__ __forceinline__", 0, text.index(name))
    assert start >= 0, name
    opening = text.index("{", text.index(name, start))
    # Skip C string literals while balancing braces: PTX has its own braces.
    depth = 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[{}]', text[opening:]):
        token = match.group()
        if token.startswith('"'):
            continue
        depth += 1 if token == "{" else -1
        if depth == 0:
            return text[start:opening + match.end()]
    raise AssertionError(f"unclosed function {name}")


def translate_asm(text, operands):
    """Translate only supported unsigned PTX; reject unfamiliar instructions."""
    asm_start = text.index("asm(")
    literal_end = text.index(":", asm_start)
    asm_end = text.index(";", text.index(");", literal_end)) + 1
    literals = re.findall(r'"(?:\\.|[^"\\])*"', text[asm_start:literal_end])
    ptx = "".join(ast.literal_eval(s) for s in literals)
    ptx = re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    declarations = []
    registers = set()

    def declare(match):
        bits, names = match.groups()
        registers.update(name.strip() for name in names.split(","))
        declarations.append(f"uint{bits}_t " + ",".join("ptx_" + name.strip() for name in names.split(",")) + ";")
        return ""

    ptx = re.sub(r"\.reg\s+\.u(32|64)\s+([^;]+);", declare, ptx)
    # Block braces are standalone; mov.b64 tuple braces stay.
    ptx = re.sub(r"(?m)^\s*[{}]\s*$", "", ptx)
    ptx = re.sub(r";\s*}", ";", ptx)
    code = ["{", *declarations, "uint32_t cc=0;", "__uint128_t tmp;"]

    def operand(name):
        name = name.strip()
        if name.startswith("%"):
            return operands[int(name[1:])]
        return "ptx_" + name if name in registers else name

    for line in ptx.split(";"):
        line = line.strip()
        if not line:
            continue
        op, args = line.split(None, 1)
        if op == "mov.b64" and "{" in args:
            split = re.fullmatch(r"\{([^,]+),([^}]+)\}\s*,\s*(.+)", args)
            join = re.fullmatch(r"([^,]+),\s*\{([^,]+),([^}]+)\}", args)
            if split:
                lo, hi, src = map(operand, split.groups())
                code += [f"{lo}=uint32_t({src}); {hi}=uint32_t(uint64_t({src})>>32);"]
            elif join:
                dst, lo, hi = map(operand, join.groups())
                code += [f"{dst}=uint64_t({lo}) | (uint64_t({hi})<<32);"]
            else:
                raise AssertionError(line)
            continue
        args = list(map(operand, args.split(",")))
        dst, src = args[:2]
        if op in ("mov.u32", "mov.u64", "mov.b64"):
            code += [f"{dst}={src};"]
        elif op == "mul.hi.u32":
            code += [f"{dst}=uint32_t((uint64_t({src})*uint64_t({args[2]}))>>32);"]
        elif op == "mul.wide.u32":
            code += [f"{dst}=uint64_t({src})*uint64_t({args[2]});"]
        elif op.startswith("add"):
            bits = int(op.rsplit("u", 1)[1])
            carry = "+cc" if op.startswith("addc") else ""
            code += [f"tmp=__uint128_t({src})+__uint128_t({args[2]}){carry};",
                     f"{dst}=uint{bits}_t(tmp);"]
            if ".cc." in op:
                code += [f"cc=uint32_t(tmp>>{bits});"]
        elif op in ("and.b32", "xor.b32"):
            symbol = "&" if op.startswith("and") else "^"
            code += [f"{dst}={src}{symbol}{args[2]};"]
        else:
            raise AssertionError(f"unsupported PTX instruction: {line}")
    code += ["}"]
    result = text[:asm_start] + "\n".join(code) + text[asm_end:]
    return result, ptx


def host_source():
    packed = preprocess(HERE / "PackedRecovery.cuh", 0)
    mult = function(preprocess(HERE / "GPUMath.h", 0), "_ModMultCore")
    mult, _ = translate_asm(mult, [f"r{i}" for i in range(4)] +
                            [f"a[{i}]" for i in range(4)] + [f"b[{i}]" for i in range(4)])
    parity = function(packed, "qsb_sum_parity")
    parity, _ = translate_asm(parity, ["s0", "s1", "s2", "s3", "c"] +
                              [f"w[{i}]" for i in range(4)] + [f"b[{i}]" for i in range(4)])
    parts = ["#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <array>\n#include <vector>",
             "#define __device__\n#define __forceinline__ inline", mult, parity,
             "void qsb_packed_raw_mul(uint64_t *r,const uint64_t *a,const uint64_t *b){_ModMultCore(r,a,b);}"]
    counts = []
    for narrow in (0, 1):
        src = preprocess((HERE / "ParityWindow.cuh" if not narrow else LAB / "ParityHighParts.cuh"), narrow)
        words = function(src, "qsb_parity_window_words")
        words, ptx = translate_asm(words, ["mid", "top"] +
                                   [f"a[{i}]" for i in range(4)] + [f"b[{i}]" for i in range(4)])
        counts.append(ptx.count("mul.wide.u32"))
        wrapper = function(src, "qsb_parity_product_window")
        for fun in (words, wrapper):
            fun = fun.replace("qsb_parity_window_words", f"words{narrow}")
            fun = fun.replace("qsb_parity_product_window", f"parity{narrow}")
            parts.append(fun)
    assert counts == [27, 9], counts
    return "\n".join(parts) + MAIN, counts


MAIN = r"""
using U128=__uint128_t;
using Row=std::array<uint64_t,4>;
static uint64_t rows=0,accepted=0,fallbacks=0,extra_fallbacks=0;
static uint64_t max_dm=0,max_dh=0;
static uint64_t rng=0x715b18aa299fcc01ULL;
static uint64_t random64(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static void need(bool ok,const char *what,const Row&a,const Row&b,const Row&beta){
    if(ok)return;
    fprintf(stderr,"FAIL row=%llu %s\n",(unsigned long long)rows,what);
    for(const Row *r:{&a,&b,&beta}){for(auto v:*r)fprintf(stderr,"%016llx ",(unsigned long long)v);fprintf(stderr,"\n");}
    exit(1);
}
struct Model {uint64_t mid,top; U128 fullmid;};
static Model model(const Row&a,const Row&b,bool narrow){
    U128 d[15]={};
    for(int i=0;i<8;i++)for(int j=0;j<8;j++)
        d[i+j]+=U128(uint32_t(a[i/2]>>(32*(i%2))))*uint32_t(b[j/2]>>(32*(j%2)));
    U128 carry6=0,carry13=0;
    for(int i=0;i<8;i++)for(int j=0;j<8;j++){
        U128 prod=U128(uint32_t(a[i/2]>>(32*(i%2))))*uint32_t(b[j/2]>>(32*(j%2)));
        if(i+j==6)carry6+=prod>>32;
        if(i+j==13)carry13+=prod>>32;
    }
    U128 m=d[7]+(narrow?carry6:((d[6]+(d[5]>>32))>>32));
    U128 h=d[14]+(narrow?carry13:((d[13]+(d[12]>>32))>>32));
    uint64_t mid=(uint64_t(m)&0x1ffffffffULL)^((uint64_t(d[8])&1)<<32);
    return {mid,uint64_t(h),m};
}
static uint64_t qvalue(uint64_t m,uint64_t t,const Row&beta){return t+977ULL*(t>>32)+uint32_t(m)+(beta[3]>>32);}
static bool guard(uint64_t m,uint64_t q,bool narrow){return narrow ? uint32_t(m)<0xfffffff3u&&uint32_t(q)<0xfffff478u : uint32_t(m)!=0xffffffffu&&uint32_t(q)<0xfffff859u;}
static void check(const Row&a,const Row&b,const Row&beta){
    uint64_t m0,h0,m1,h1;
    words0(m0,h0,a.data(),b.data());words1(m1,h1,a.data(),b.data());
    auto ref0=model(a,b,false),ref1=model(a,b,true);
    need(m0==ref0.mid&&h0==ref0.top,"original PTX versus independent columns",a,b,beta);
    need(m1==ref1.mid&&h1==ref1.top,"narrow PTX versus independent columns",a,b,beta);
    need(ref0.fullmid>=ref1.fullmid&&ref0.fullmid-ref1.fullmid<=12,"mid carry bound",a,b,beta);
    need(h0>=h1&&h0-h1<=4,"top carry bound",a,b,beta);
    uint64_t dm=uint64_t(ref0.fullmid-ref1.fullmid),dh=h0-h1;
    if(dm>max_dm)max_dm=dm;if(dh>max_dh)max_dh=dh;
    uint64_t q0=qvalue(m0,h0,beta),q1=qvalue(m1,h1,beta);
    bool g0=guard(m0,q0,false),g1=guard(m1,q1,true);
    if(g1){
        need(g0,"new fast implies inherited fast",a,b,beta);
        need(((m0^m1)&(1ULL<<32))==0,"mid parity bit",a,b,beta);
        need(((q0^q1)&(1ULL<<32))==0,"q parity bit",a,b,beta);
        accepted++;
    } else {fallbacks++;if(g0)extra_fallbacks++;}
    uint64_t raw[4];_ModMultCore(raw,a.data(),b.data());
    for(uint32_t neg:{0u,1u}){
        uint32_t p0=parity0(a.data(),b.data(),beta.data(),neg);
        uint32_t p1=parity1(a.data(),b.data(),beta.data(),neg);
        uint32_t exact=qsb_sum_parity(raw,beta.data(),neg);
        need(p0==p1,"original versus narrow full result",a,b,beta);
        need(p1==exact,"narrow versus translated device fallback",a,b,beta);
    }
    rows++;
}
static void qedges(const Row&a,const Row&b){
    uint64_t m,h;words1(m,h,a.data(),b.data());
    uint64_t base=h+977ULL*(h>>32)+uint32_t(m);
    for(uint32_t target:{0u,1u,0xfffff477u,0xfffff478u,0xfffff479u,0xfffff858u,0xfffff859u,0xfffff85au,0xfffffffeu,0xffffffffu}){
        Row beta={random64(),random64(),random64(),uint64_t(uint32_t(target-uint32_t(base)))<<32};
        check(a,b,beta);
    }
}
int main(){
    std::vector<Row> directed;
    for(uint64_t p:{0ULL,1ULL,0xffffffffULL,0x100000000ULL,0x8000000080000000ULL,0xfffffffffffffffeULL,0xffffffffffffffffULL,0xaaaaaaaaaaaaaaaaULL,0x5555555555555555ULL,0xffff0001ffff0001ULL})directed.push_back({p,p,p,p});
    for(int bit=0;bit<256;bit++){Row a={};a[bit/64]=1ULL<<(bit%64);directed.push_back(a);}
    for(auto&a:directed)for(auto&b:directed)check(a,b,Row{1,2,3,4});
    for(size_t i=0;i<9;i++)for(size_t j=0;j<9;j++)qedges(directed[i],directed[j]);
    // a0=1 lets b7 directly target the narrowed mid's low word without
    // changing D6. Exercise both sides of the x7 guards, then q guards.
    for(int rep=0;rep<1000;rep++){
        Row a={random64(),random64(),random64(),random64()};
        Row b={random64(),random64(),random64(),random64()};
        a[0]=(a[0]&0xffffffff00000000ULL)|1;b[3]&=0xffffffffULL;
        uint64_t m,h;words1(m,h,a.data(),b.data());
        for(uint32_t target:{0u,1u,0xfffffff1u,0xfffffff2u,0xfffffff3u,0xfffffff4u,0xfffffffeu,0xffffffffu}){
            b[3]=(b[3]&0xffffffffULL)|(uint64_t(uint32_t(target-uint32_t(m)))<<32);
            qedges(a,b);
        }
    }
    for(int i=0;i<2000000;i++){
        Row a,b,beta;for(int j=0;j<4;j++){a[j]=random64();b[j]=random64();beta[j]=random64();}
        check(a,b,beta);
        if(i<1000)qedges(a,b);
    }
    printf("{\"rows\":%llu,\"new_fast\":%llu,\"fallback\":%llu,\"additional_fallback\":%llu,\"max_mid_delta\":%llu,\"max_top_delta\":%llu,\"parity_comparisons\":%llu}\n",
        (unsigned long long)rows,(unsigned long long)accepted,(unsigned long long)fallbacks,(unsigned long long)extra_fallbacks,
        (unsigned long long)max_dm,(unsigned long long)max_dh,(unsigned long long)(rows*4));
}
"""


def main():
    src, counts = host_source()
    with tempfile.TemporaryDirectory(prefix="qsb-parity-cpu-") as tmp:
        cpp, exe = Path(tmp) / "test.cpp", Path(tmp) / "test"
        cpp.write_text(src)
        subprocess.run(["g++", "-std=c++17", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(exe)], check=True)
        result = json.loads(subprocess.check_output([str(exe)], text=True))
    result.update(test="extracted integer PTX semantics: window, device multiplier and sum parity",
                  wide_products_original_window=counts[0], wide_products_experiment=counts[1],
                  high_products_experiment=9, total_products_current_and_experiment=18,
                  cuda_compiled=False, gpu_executed=False, speedup=None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
