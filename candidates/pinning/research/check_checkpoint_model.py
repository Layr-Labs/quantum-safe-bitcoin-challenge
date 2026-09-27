#!/usr/bin/env python3
"""Independent model of a proposed 62-node checkpoint, not a CUDA test.

Four neighboring leaves form each quartet. Persist only nodes 2..63 of the
64-leaf heap; finish reconstructs quartet products from saved denominator
leaves. Identity substitution happens before both passes. This changes no
production source and establishes neither GPU code generation nor speed.
"""
import json
import random

P = (1 << 256) - (1 << 32) - 977


def proposed_inverse(values):
    counts = {"prepare_multiply": 0, "finish_recompute_multiply": 0,
              "finish_expand_multiply": 0}
    def mul(a, b, kind):
        counts[kind] += 1
        return a*b % P
    tree = [0]*128
    pairs = []
    for q in range(64):
        a,b,c,d = values[4*q:4*q+4]
        l=mul(a,b,"prepare_multiply");r=mul(c,d,"prepare_multiply")
        tree[64+q]=mul(l,r,"prepare_multiply")
    for node in range(63,0,-1):
        tree[node]=mul(tree[2*node],tree[2*node+1],"prepare_multiply")
    checkpoint = tree[2:64]
    root_inverse=pow(tree[1],-1,P)
    assert len(checkpoint)==62
    restored=[0]*128
    restored[1]=root_inverse;restored[2:64]=checkpoint
    for q in range(64):
        a,b,c,d=values[4*q:4*q+4]
        l=mul(a,b,"finish_recompute_multiply");r=mul(c,d,"finish_recompute_multiply")
        pairs.append((l,r))
        restored[64+q]=mul(l,r,"finish_recompute_multiply")
    for width in (1,2,4,8,16,32):
        for node in range(width,2*width):
            inv,l,r=restored[node],restored[2*node],restored[2*node+1]
            restored[2*node]=mul(inv,r,"finish_expand_multiply")
            restored[2*node+1]=mul(inv,l,"finish_expand_multiply")
    out=[]
    for q,(l,r) in enumerate(pairs):
        li=mul(restored[64+q],r,"finish_expand_multiply")
        ri=mul(restored[64+q],l,"finish_expand_multiply")
        a,b,c,d=values[4*q:4*q+4]
        out += [mul(li,b,"finish_expand_multiply"),mul(li,a,"finish_expand_multiply"),
                mul(ri,d,"finish_expand_multiply"),mul(ri,c,"finish_expand_multiply")]
    return out,counts


def main():
    rng=random.Random(260916620)
    cases=[]
    for n in (0,1,3,4,31,32,33,63,64,65,127,128,129,255,256):
        cases.append(([rng.randrange(P) for _ in range(256)],[i<n for i in range(256)]))
    cases.append(([0,1,P-1,P,P+1,P-65537,2,3]*32,[True]*256))
    for _ in range(100):
        cases.append(([rng.randrange(P) for _ in range(256)],[bool(rng.randrange(8)) for _ in range(256)]))
    for raw,active in cases:
        factors=[v%P if use and v%P else 1 for v,use in zip(raw,active)]
        out,counts=proposed_inverse(factors)
        assert out==[pow(v,-1,P) for v in factors]
        assert counts=={"prepare_multiply":255,"finish_recompute_multiply":192,"finish_expand_multiply":510}
    print(json.dumps({"status":"PASS","validation_level":"independent_python_model",
        "blocks":len(cases),"leaf_inverses":256*len(cases),"checkpoint_nodes":62,
        "counts_per_block":counts,"extra_multiply_per_candidate":192/256,
        "checkpoint_allocation_mib_16M":{"base":512,"proposed":128},
        "checkpoint_write_plus_read_saved_mib_16M":768,
        "gpu_executed":False,"production_modified":False},indent=2))


if __name__=="__main__":main()
