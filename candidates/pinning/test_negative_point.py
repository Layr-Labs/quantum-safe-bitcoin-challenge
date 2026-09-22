#!/usr/bin/env python3
"""Compile the actual negative-Y point formulas against exact field adapters.

The production header and the positive-Y GPUMath.h functions are extracted,
not transcribed. OpenSSL supplies an independent secp256k1 point-add oracle.
This audits the coordinate convention, signed offset table inputs and final
ordinate restoration. It does NOT execute CUDA/PTX or certify the inherited
short-carry field reduction. Requires g++, Boost headers and libcrypto.
SPDX-License-Identifier: GPL-3.0-only
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def function_body(source, name):
    match = re.search(r"\bvoid\s+" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source)
    if not match:
        raise AssertionError(f"Function definition not found: {name}")
    end = match.end()
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


ADAPTERS = r'''
#include <boost/multiprecision/cpp_int.hpp>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using boost::multiprecision::cpp_int;
using Field = std::array<uint64_t,4>;
const cpp_int B = cpp_int(1)<<256;
const cpp_int K = (cpp_int(1)<<32)+977;
const cpp_int PRIME = B-K;
const cpp_int OFFSET = (K-1)/2;
const cpp_int MASK64 = (cpp_int(1)<<64)-1;
cpp_int readf(const uint64_t *a) {
    cpp_int x=0;
    for(int i=3;i>=0;i--) { x<<=64; x+=a[i]; }
    return x;
}
cpp_int mod(cpp_int x) { x%=PRIME; if(x<0)x+=PRIME; return x; }
void rawwrite(uint64_t *r,cpp_int x) {
    for(int i=0;i<4;i++) { r[i]=(x&MASK64).convert_to<uint64_t>(); x>>=64; }
}
void writef(uint64_t *r,cpp_int x) { rawwrite(r,mod(x)); }
void Load256(uint64_t *r,const uint64_t *a) { std::memmove(r,a,32); }
void _ModMult(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    writef(r,readf(a)*readf(b));
}
void _ModMult(uint64_t *r,const uint64_t *b) { _ModMult(r,r,b); }
void _ModSqr(uint64_t *r,const uint64_t *a) { writef(r,readf(a)*readf(a)); }
void _ModSub256(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    writef(r,readf(a)-readf(b));
}
void _ModSub256(uint64_t *r,const uint64_t *b) { _ModSub256(r,r,b); }
void _ModAdd256(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    writef(r,readf(a)+readf(b));
}
void _ModAdd256(uint64_t *r,const uint64_t *b) { _ModAdd256(r,r,b); }
void _ModAddLazy(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    _ModAdd256(r,a,b);
}
void _ModAddLazyOff(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    writef(r,readf(a)+readf(b)-(K-1));
}
void _ModNeg256(uint64_t *r,const uint64_t *a) { writef(r,-readf(a)); }
void _ModX3Fused(uint64_t *r,const uint64_t *a,const uint64_t *e,const uint64_t *q) {
    writef(r,readf(a)+readf(e)-2*readf(q));
}
void _ModSqrAddSub2(uint64_t *r,const uint64_t *a,const uint64_t *e,const uint64_t *q) {
    writef(r,readf(a)*readf(a)+readf(e)-2*readf(q));
}
void qsb_muladd_seed(uint64_t *r,const uint64_t *a,const uint64_t *b,const uint64_t *c) {
    writef(r,readf(a)*readf(b)+readf(c));
}
void require(bool yes,const char *message) {
    if(!yes)throw std::runtime_error(message);
}
Field frombn(const BIGNUM *bn) {
    unsigned char bytes[32];
    require(BN_bn2lebinpad(bn,bytes,32)==32,"BN encoding failed");
    Field r{};
    for(int i=0;i<32;i++)r[i/8]|=uint64_t(bytes[i])<<(8*(i%8));
    return r;
}
void tobn(BIGNUM *bn,const Field &a) {
    unsigned char bytes[32];
    for(int i=0;i<32;i++)bytes[i]=(unsigned char)(a[i/8]>>(8*(i%8)));
    require(BN_lebin2bn(bytes,32,bn)!=nullptr,"BN decoding failed");
}
struct Affine { Field x,y; };
struct XYZZ { Field x,y,u,v; };
Affine affine(const EC_GROUP *group,const EC_POINT *point,BN_CTX *ctx) {
    BIGNUM *x=BN_new(),*y=BN_new();
    require(EC_POINT_get_affine_coordinates(group,point,x,y,ctx)==1,"Affine extraction failed");
    Affine p{frombn(x),frombn(y)};
    BN_free(x); BN_free(y); return p;
}
void setpoint(const EC_GROUP *group,EC_POINT *point,const Affine &a,BN_CTX *ctx) {
    BIGNUM *x=BN_new(),*y=BN_new(); tobn(x,a.x); tobn(y,a.y);
    require(EC_POINT_set_affine_coordinates(group,point,x,y,ctx)==1,"Point load failed");
    BN_free(x); BN_free(y);
}
Field table_y(const Affine &unsigned_p,bool negative) {
    Field y;
#if QSB_YOFF
    rawwrite(y.data(),readf(unsigned_p.y.data())+OFFSET);
    if(negative)for(auto &limb:y)limb=~limb;
    cpp_int expected=negative ? PRIME-readf(unsigned_p.y.data()) : readf(unsigned_p.y.data());
    require(readf(y.data())==expected+OFFSET,"Signed table XOR/offset identity");
#else
    writef(y.data(),negative ? -readf(unsigned_p.y.data()) : readf(unsigned_p.y.data()));
#endif
    return y;
}
Field actual_anchor(const Field &stored) {
    Field y;
#if QSB_YOFF
    writef(y.data(),readf(stored.data())-OFFSET);
#else
    y=stored;
#endif
    return y;
}
void check(const XYZZ &negative,const XYZZ &positive,const Field &anchor,
           const Affine &reference) {
    require(negative.x==positive.x && negative.u==positive.u && negative.v==positive.v,
            "Positive/negative XYZZ scale or X mismatch");
    require(mod(readf(negative.y.data())+readf(positive.y.data()))==0,
            "Deferred ordinates are not additive inverses");
    const cpp_int u=readf(negative.u.data()),v=readf(negative.v.data());
    require(u!=0 && v!=0,"Unexpected exceptional addition");
    require(mod(v*v-u*u*u)==0,"XYZZ scale identity");
    const Field actual=actual_anchor(anchor);
    const cpp_int y=mod(-readf(negative.y.data())-readf(actual.data())*v);
    require(readf(negative.x.data())==mod(readf(reference.x.data())*u),
            "X disagrees with OpenSSL point addition");
    require(y==mod(readf(reference.y.data())*v),
            "Resolved Y disagrees with OpenSSL point addition");
}
void rescale(XYZZ &p,const cpp_int &s) {
    const cpp_int s2=mod(s*s),s3=mod(s2*s);
    writef(p.x.data(),readf(p.x.data())*s2);
    writef(p.u.data(),readf(p.u.data())*s2);
    writef(p.y.data(),readf(p.y.data())*s3);
    writef(p.v.data(),readf(p.v.data())*s3);
}
'''


DRIVER = r'''
int main() {
    try {
        std::mt19937_64 rng(0x4e45475953454544ULL);
        EC_GROUP *group=EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx=BN_CTX_new();
        EC_POINT *tmp=EC_POINT_new(group),*sum=EC_POINT_new(group);
        BIGNUM *scalar=BN_new(),*order=BN_new();
        require(group && ctx && tmp && sum && scalar && order,"OpenSSL allocation");
        require(EC_GROUP_get_order(group,order,ctx)==1,"Group order");
        std::vector<Affine> pool;
        for(unsigned i=0;i<256;i++) {
            Field words{rng(),rng(),rng(),rng()}; tobn(scalar,words);
            if(i<8)require(BN_set_word(scalar,i+1)==1,"Small scalar");
            else if(i<16) { require(BN_copy(scalar,order)!=nullptr,"Order copy"); BN_sub_word(scalar,i-7); }
            require(EC_POINT_mul(group,tmp,scalar,nullptr,nullptr,ctx)==1,"Independent scalar multiply");
            pool.push_back(affine(group,tmp,ctx));
        }
        uint64_t checks=0,signed_loads=0,scaled=0;
        for(unsigned trial=0;trial<2048;trial++) {
            std::array<Affine,15> points;
            std::array<Field,15> encoded_y;
            Affine current{};
            for(unsigned j=0;j<15;j++) {
                for(;;) {
                    const Affine &p=pool[rng()%pool.size()];
                    bool negative=(rng()&1)!=0;
                    points[j]=p;
                    if(negative)writef(points[j].y.data(),-readf(p.y.data()));
                    if(j && points[j].x==current.x)continue;
                    encoded_y[j]=table_y(p,negative); signed_loads++;
                    setpoint(group,tmp,points[j],ctx);
                    if(j==0)require(EC_POINT_copy(sum,tmp)==1,"Point copy");
                    else require(EC_POINT_add(group,sum,sum,tmp,ctx)==1,"Independent point addition");
                    require(!EC_POINT_is_at_infinity(group,sum),"Unexpected infinity");
                    current=affine(group,sum,ctx); break;
                }
            }
            XYZZ n{},p{};
            CANDIDATE_SEED(n.x.data(),n.y.data(),n.u.data(),n.v.data(),
                points[0].x.data(),encoded_y[0].data(),points[1].x.data(),encoded_y[1].data());
            _PointAddXYZZ_mm(p.x.data(),p.y.data(),p.u.data(),p.v.data(),
                points[0].x.data(),encoded_y[0].data(),points[1].x.data(),encoded_y[1].data());
            setpoint(group,sum,points[0],ctx); setpoint(group,tmp,points[1],ctx);
            require(EC_POINT_add(group,sum,sum,tmp,ctx)==1,"Seed reference sum");
            Field anchor=encoded_y[0];
            check(n,p,anchor,affine(group,sum,ctx)); checks++;
            for(unsigned j=2;j<15;j++) {
                if((j+trial)%4==0) {
                    cpp_int scale;
                    if(trial%7==0)scale=PRIME-1;
                    else { Field words{rng(),rng(),rng(),rng()}; scale=mod(readf(words.data())); if(scale==0)scale=1; }
                    rescale(n,scale);rescale(p,scale);scaled++;
                }
                CANDIDATE_ADD(n.x.data(),n.y.data(),n.u.data(),n.v.data(),
                    points[j].x.data(),encoded_y[j].data(),anchor.data());
                _PointAddXYZZT<true>(p.x.data(),p.y.data(),p.u.data(),p.v.data(),
                    points[j].x.data(),encoded_y[j].data(),anchor.data());
                anchor=encoded_y[j];
                setpoint(group,tmp,points[j],ctx);
                require(EC_POINT_add(group,sum,sum,tmp,ctx)==1,"Chain reference sum");
                check(n,p,anchor,affine(group,sum,ctx)); checks++;
            }
            // Execute the same two function calls as the scalar's production tail.
            uint64_t Y[4],V[4],y0[4],x1[4];
            Load256(Y,n.y.data());Load256(V,n.v.data());
            Field true_anchor=actual_anchor(anchor);Load256(y0,true_anchor.data());
            PRODUCTION_RESOLVE
            Field old_actual;
            writef(old_actual.data(),readf(p.y.data())-readf(true_anchor.data())*readf(p.v.data()));
            require(readf(Y)==readf(old_actual.data()),"Production final Y restoration");
        }
        std::cout << "PASS: 2048 chains, " << checks << " intermediate states, "
                  << signed_loads << " signed table loads, " << scaled
                  << " projective rescalings; exact-field adapters, no GPU/PTX\n";
        BN_free(scalar); BN_free(order);EC_POINT_free(tmp);EC_POINT_free(sum);
        BN_CTX_free(ctx);EC_GROUP_free(group);
    } catch(const std::exception &e) { std::cerr<<e.what()<<"\n";return 1; }
}
'''


def main():
    header = HERE / "NegativePoint.cuh"
    point = header.read_text()
    math = (HERE / "GPUMath.h").read_text()
    scalar = (HERE / "pinning.cu").read_text()
    add_name = re.findall(r"\bvoid\s+(\w+_point_add)\s*\(", point)
    seed_name = re.findall(r"\bvoid\s+(\w+_point_seed)\s*\(", point)
    assert len(add_name) == len(seed_name) == 1
    resolve = re.search(
        r"qsb_muladd_seed\(x1,\s*y0,\s*V,\s*Y\);\s*_ModNeg256\(Y,\s*x1\);", scalar
    )
    assert resolve, "Production negative-Y restoration was not found; re-audit the integration"
    bodies = "\n".join([
        function_body(point, add_name[0]), function_body(point, seed_name[0]),
        "template<bool DEFER_Y>\n" + function_body(math, "_PointAddXYZZT"),
        function_body(math, "_PointAddXYZZ_mm"),
    ])
    driver = DRIVER.replace("CANDIDATE_ADD", add_name[0]).replace("CANDIDATE_SEED", seed_name[0])
    driver = driver.replace("PRODUCTION_RESOLVE", resolve.group())
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for the actual-header CPU audit"
    with tempfile.TemporaryDirectory(prefix="qsb-negative-point-") as directory:
        cpp = Path(directory) / "audit.cpp"
        binary = Path(directory) / "audit"
        cpp.write_text(ADAPTERS + "\n" + bodies + "\n" + driver)
        # Production defaults and the non-offset/non-fused formula fallback.
        for offset, fused in [(1, 1), (0, 0)]:
            subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                f"-DQSB_YOFF={offset}", f"-DQSB_FUSE_SQRADDSUB2={fused}",
                "-DQSB_LAZY=1", str(cpp), "-lcrypto", "-o", str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)
        # A wrong deferred sign must fail against both the inherited chain
        # and OpenSSL. This mutation changes the actual extracted add body.
        wrong = function_body(point, add_name[0]).replace(
            "_ModSub256(Q, T, Q);", "_ModSub256(Q, Q, T);", 1
        )
        assert wrong != function_body(point, add_name[0]), "Negative-control mutation did not apply"
        mutated = bodies.replace(function_body(point, add_name[0]), wrong, 1)
        cpp.write_text(ADAPTERS + "\n" + mutated + "\n" + driver)
        subprocess.run([
            compiler, "-std=c++17", "-O2", "-DQSB_YOFF=1",
            "-DQSB_FUSE_SQRADDSUB2=1", "-DQSB_LAZY=1", str(cpp),
            "-lcrypto", "-o", str(binary),
        ], check=True)
        negative = subprocess.run([str(binary)], capture_output=True, text=True)
        assert negative.returncode != 0 and "Deferred ordinates" in negative.stderr, negative
        print("PASS: deliberately reversed deferred-Y sign is rejected")


if __name__ == "__main__":
    main()
