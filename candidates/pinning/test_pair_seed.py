#!/usr/bin/env python3
"""Actual-header CPU audit of PairSeed.cuh using exact field adapters.

OpenSSL verifies the regrouped 15-point sum, including random curve
isomorphisms. The actual product-tree body executes with CPU threads and
barriers. Only CUDA plumbing, field primitives and the offset-add PTX use
host adapters; this is not a GPU performance or C31-reduction test.
Requires g++ (C++20), Boost headers and libcrypto.
SPDX-License-Identifier: GPL-3.0-only
"""

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("negative_point_audit", HERE / "test_negative_point.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

EXTRA = r'''
#include <barrier>
#include <thread>
#define __shared__ static
struct ThreadIndex { unsigned x; };
thread_local ThreadIndex threadIdx;
std::barrier<> *cta_barrier;
void __syncthreads() { cta_barrier->arrive_and_wait(); }
void qsb_field_normalize(uint64_t *a) { writef(a,readf(a)); }
void qsb_field_mul(uint64_t *r,const uint64_t *a,const uint64_t *b) {
    _ModMult(r,a,b);r[4]=0;
}
void _ModInv(uint64_t *a) {
    const cpp_int value=mod(readf(a));
    require(value!=0,"Unexpected inversion of zero");
    writef(a,boost::multiprecision::powm(value,PRIME-2,PRIME));a[4]=0;
}
void qsb_yoff_to_y(uint64_t *y) { writef(y,readf(y)-OFFSET); }
void qsb_decode_to_shared(const uint64_t *) {}
unsigned gt_offset(unsigned c) { return c; }
void qsb_load_decoded(const uint8_t *table,unsigned c,unsigned,uint64_t *x,uint64_t *y) {
    const Affine *points=reinterpret_cast<const Affine*>(table);
    Load256(x,points[c].x.data());Load256(y,points[c].y.data());
}
unsigned fallback_calls=0;
void _FixedBaseSignedXYZZScalar(uint64_t*,uint64_t*,uint64_t*,uint64_t*,const uint64_t*,const uint8_t*);
template<int N> void qsb_pair_product_inverse(uint64_t *out,const uint64_t *a,
                                            const uint64_t *b,bool singular) {
    (void)N;
    if(singular) { writef(out,1);return; }
    uint64_t product[5];qsb_field_mul(product,a,b);_ModInv(product);Load256(out,product);
}
'''

DRIVER = r'''
void _FixedBaseSignedXYZZScalar(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,
                               const uint64_t*,const uint8_t *table) {
    fallback_calls++;
    const Affine *p=reinterpret_cast<const Affine*>(table);
    _PointAddXYZZ_mm(X,Y,U,V,p[0].x.data(),p[0].y.data(),p[1].x.data(),p[1].y.data());
    Field anchor=p[0].y;
    for(unsigned j=2;j<15;j++) {
        _PointAddXYZZT<true>(X,Y,U,V,p[j].x.data(),p[j].y.data(),anchor.data());
        anchor=p[j].y;
    }
    const Field a=actual_anchor(anchor);
    uint64_t t[4];_ModMult(t,a.data(),V);_ModSub256(Y,Y,t);
}
template<int N> void audit_tree(std::mt19937_64 &rng) {
    for(unsigned mode=0;mode<4;mode++) {
        std::array<Field,N> a,b,output;
        std::array<bool,N> singular{};
        for(unsigned j=0;j<N;j++) {
            rawwrite(a[j].data(),cpp_int(rng())+1);
            rawwrite(b[j].data(),cpp_int(rng())+1);
            singular[j]=(mode==1 && j%17==0) || (mode==2 && j==N/2) || mode==3;
            if(singular[j]) {
                // Zero in either slot must be isolated before the tree root.
                if(j&1)writef(a[j].data(),0);else writef(b[j].data(),0);
            }
        }
        std::barrier barrier(N);cta_barrier=&barrier;
        std::vector<std::thread> threads;
        for(unsigned j=0;j<N;j++)threads.emplace_back([&,j] {
            threadIdx.x=j;
            actual_pair_product_inverse<N>(output[j].data(),a[j].data(),b[j].data(),singular[j]);
        });
        for(auto &thread:threads)thread.join();
        for(unsigned j=0;j<N;j++) {
            if(singular[j])require(readf(output[j].data())==1,"Singular tree identity leaf");
            else require(mod(readf(output[j].data())*readf(a[j].data())*readf(b[j].data()))==1,
                         "Product-tree inverse differs from independent identity");
        }
    }
}
int main() {
    try {
        std::mt19937_64 rng(0x5041495253454544ULL);
        audit_tree<128>(rng);audit_tree<256>(rng);
        const std::vector<cpp_int> edges{0,1,2,K-1,K,PRIME-2,PRIME-1};
        unsigned boundary_count=0;
        for(const cpp_int &input_x:edges)for(const cpp_int &output_x:edges)for(const cpp_int &output_y:edges) {
            Field x,y,dx,slope;
            writef(x.data(),input_x);
            writef(y.data(),input_x-output_x-output_y);
#if QSB_YOFF
            rawwrite(y.data(),readf(y.data())+OFFSET);
#endif
            writef(dx.data(),1-2*input_x-output_x);writef(slope.data(),1);
            qsb_pair_affine_finish(x.data(),y.data(),dx.data(),slope.data());
            require(readf(x.data())==output_x,"Affine-finish X boundary");
#if QSB_YOFF
            require(readf(y.data())==output_y+OFFSET,"Canonical offset-Y boundary");
#else
            require(readf(y.data())==output_y,"Canonical Y boundary");
#endif
            boundary_count++;
        }
        EC_GROUP *group=EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx=BN_CTX_new();EC_POINT *tmp=EC_POINT_new(group),*sum=EC_POINT_new(group);
        BIGNUM *scalar=BN_new();std::vector<Affine> pool;
        for(unsigned i=0;i<128;i++) {
            Field s{rng(),rng(),rng(),rng()};tobn(scalar,s);
            if(i<8)BN_set_word(scalar,i+1);
            require(EC_POINT_mul(group,tmp,scalar,nullptr,nullptr,ctx)==1,"Pool scalar multiplication");
            pool.push_back(affine(group,tmp,ctx));
        }
        unsigned singular_cases=0,chains=0;
        for(unsigned trial=0;trial<512;trial++) {
            Field random_scale{rng(),rng(),rng(),rng()};
            cpp_int iso=trial%4==0 ? cpp_int(1) : mod(readf(random_scale.data()));
            if(iso==0)iso=1;
            const cpp_int iso2=mod(iso*iso),iso3=mod(iso2*iso);
            std::array<Affine,15> original,table;
            for(unsigned j=0;j<15;j++) {
                original[j]=pool[rng()%pool.size()];
                if(rng()&1)writef(original[j].y.data(),-readf(original[j].y.data()));
                // Test the second-pair singular fallback while the original
                // serial chain remains well-defined on these random sums.
                if(trial%16==0 && j==3)original[j]=original[2];
                writef(table[j].x.data(),readf(original[j].x.data())*iso2);
                writef(table[j].y.data(),readf(original[j].y.data())*iso3);
#if QSB_YOFF
                rawwrite(table[j].y.data(),readf(table[j].y.data())+OFFSET);
#endif
                setpoint(group,tmp,original[j],ctx);
                if(j==0)EC_POINT_copy(sum,tmp);else EC_POINT_add(group,sum,sum,tmp,ctx);
            }
            if(EC_POINT_is_at_infinity(group,sum))continue;
            XYZZ paired{},serial{};const Field dummy{};
            const unsigned before=fallback_calls;
            if(trial&16)
                qsb_pairseed_scalar<false>(paired.x.data(),paired.y.data(),paired.u.data(),paired.v.data(),
                                          dummy.data(),reinterpret_cast<const uint8_t*>(table.data()));
            else
                qsb_pairseed_scalar<true>(paired.x.data(),paired.y.data(),paired.u.data(),paired.v.data(),
                                         dummy.data(),reinterpret_cast<const uint8_t*>(table.data()));
            if(fallback_calls!=before)singular_cases++;
            _FixedBaseSignedXYZZScalar(serial.x.data(),serial.y.data(),serial.u.data(),serial.v.data(),
                                      dummy.data(),reinterpret_cast<const uint8_t*>(table.data()));
            if(readf(serial.u.data())==0 || readf(serial.v.data())==0) {
                require(paired.x==serial.x && paired.y==serial.y && paired.u==serial.u && paired.v==serial.v,
                        "Inherited exceptional serial result changed");
                continue;
            }
            require(readf(paired.u.data())!=0 && readf(paired.v.data())!=0,"Pair regrouping produced singular state");
            require(mod(readf(paired.x.data())*readf(serial.u.data())-readf(serial.x.data())*readf(paired.u.data()))==0,
                    "Paired/serial X differs");
            require(mod(readf(paired.y.data())*readf(serial.v.data())-readf(serial.y.data())*readf(paired.v.data()))==0,
                    "Paired/serial restored Y differs");
            const Affine expected=affine(group,sum,ctx);
            require(readf(paired.x.data())==mod(readf(expected.x.data())*iso2*readf(paired.u.data())),
                    "Paired X differs from independent OpenSSL sum under isomorphism");
            require(readf(paired.y.data())==mod(readf(expected.y.data())*iso3*readf(paired.v.data())),
                    "Paired Y differs from independent OpenSSL sum under isomorphism");
            require(mod(readf(paired.v.data())*readf(paired.v.data())-
                    readf(paired.u.data())*readf(paired.u.data())*readf(paired.u.data()))==0,"Paired scale identity");
            chains++;
        }
        require(chains>450 && singular_cases>=32,"Insufficient normal/fallback coverage");
        std::cout<<"PASS: "<<chains<<" regrouped chains, "<<singular_cases<<" singular fallbacks, "
                 <<boundary_count<<" affine boundaries; reload/retain anchors; 128/256-lane actual-body tree with zero isolation\n";
        BN_free(scalar);EC_POINT_free(tmp);EC_POINT_free(sum);BN_CTX_free(ctx);EC_GROUP_free(group);
    } catch(const std::exception &error) { std::cerr<<error.what()<<"\n";return 1; }
}
'''


def main():
    source = (HERE / "PairSeed.cuh").read_text()
    math = (HERE / "GPUMath.h").read_text()
    finish = AUDIT.function_body(source, "qsb_pair_affine_finish")
    assembly = re.search(r'\basm\(.*?\);', finish, re.S)
    assert assembly and assembly.group().count("addc") == 3
    assert "0x800001E8" in assembly.group()
    # This replacement models the four-limb add-with-carry, not field add.
    finish = finish[:assembly.start()] + (
        '{ cpp_int offset_sum=readf(y)+OFFSET; require(offset_sum<B,"Offset must fit 256 bits"); '
        'rawwrite(y,offset_sum); }'
    ) + finish[assembly.end():]
    tree = AUDIT.function_body(source, "qsb_pair_product_inverse")
    tree = tree.replace("qsb_pair_product_inverse", "actual_pair_product_inverse", 1)
    bodies = "\n".join([
        "template<int N>\n" + tree, finish,
        "template<bool RELOAD_ANCHORS=true>\n" + AUDIT.function_body(source, "qsb_pairseed_scalar"),
    ])
    positive = "\n".join([
        "template<bool DEFER_Y>\n" + AUDIT.function_body(math, "_PointAddXYZZT"),
        AUDIT.function_body(math, "_PointAddXYZZ_mm"),
    ])
    compiler = shutil.which("g++")
    assert compiler, "g++ is required"
    with tempfile.TemporaryDirectory(prefix="qsb-pair-seed-") as directory:
        cpp, binary = Path(directory) / "audit.cpp", Path(directory) / "audit"
        cpp.write_text(AUDIT.ADAPTERS + EXTRA + positive + bodies + DRIVER)
        for offset in (1, 0):
            subprocess.run([
                compiler, "-std=c++20", "-O2", "-pthread", "-Wno-unknown-pragmas",
                f"-DQSB_YOFF={offset}", "-DQSB_FUSE_SQRADDSUB2=1", "-DQSB_LAZY=1",
                "-DQSB_TREE_N=128", "-DGT_CHUNKS=15", str(cpp), "-lcrypto", "-o", str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)
        # Reducing y+c modulo p destroys the exact signed-table encoding.
        # Directed near-p y values above must detect that subtle mutation.
        mutated = (AUDIT.ADAPTERS + EXTRA + positive + bodies + DRIVER).replace(
            "rawwrite(y,offset_sum);", "writef(y,offset_sum);", 1
        )
        assert mutated != AUDIT.ADAPTERS + EXTRA + positive + bodies + DRIVER
        cpp.write_text(mutated)
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-pthread", "-Wno-unknown-pragmas",
            "-DQSB_YOFF=1", "-DQSB_FUSE_SQRADDSUB2=1", "-DQSB_LAZY=1",
            "-DQSB_TREE_N=128", "-DGT_CHUNKS=15", str(cpp), "-lcrypto", "-o", str(binary),
        ], check=True)
        negative = subprocess.run([str(binary)], capture_output=True, text=True)
        assert negative.returncode != 0 and "Canonical offset-Y boundary" in negative.stderr, negative
        print("PASS: incorrect modulo-p reduction of y+c is rejected")


if __name__ == "__main__":
    main()
