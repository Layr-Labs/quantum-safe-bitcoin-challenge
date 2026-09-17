#!/usr/bin/env python3
"""Integer contracts and actual source PTX semantic checks; no candidate edits."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from preflight import source_identity
from ptx_field_model import Program,extract_ptx,function

B=1<<256
C=(1<<32)+977
P=B-C
MASK=B-1
SOURCE=ROOT/'research/specialist_warps/resident_tables/packed_digits/scoped_load/candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def words(x):return [(x>>(64*i))&((1<<64)-1) for i in range(4)]
def integer(w):return sum(v<<(64*i) for i,v in enumerate(w))
def norm(a):return a-P if a>=P else a

def weak_add(a,b,B=B,C=C):
    total=a+b;u=total%B;carry=total//B
    first=u+C*carry;v=first%B;carry2=first//B
    if carry2:assert v<C and v+C<2*C
    return v+C*carry2,carry2

def weak_sub(a,b,B=B,C=C):
    borrow=int(a<b);u=(a-b)%B
    first=u-C*borrow;borrow2=int(first<0);v=first%B
    if borrow2:assert B-C<=v<B and v-C>=0
    return v-C*borrow2,borrow2

def weak_neg(a):
    t=P-a;v=t%B;borrow=int(t<0)
    if borrow:assert (v&((1<<64)-1))>=C
    return v-C*borrow

def weak_product(a,b):
    product=a*b
    first=product%B+C*(product//B)
    assert first//B<=C
    second=first%B+C*(first//B)
    assert second<=B-1+C*C
    if second>=B:assert second-B+C<C*C+C<1<<96
    return second%B+C*(second//B),int(second>=B)

def old_add(a,b):
    t=a+b
    return (t-P if t>=P else t)%B
def old_sub(a,b):return (a-b+(P if a<b else 0))%B
def old_neg(a):return (P-a)%B

def main():
    identity=source_identity(SOURCE)
    assert identity['source_fingerprint']=='cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667'
    math=(SOURCE/'GPUMath.h').read_text();square=(SOURCE/'square32.cuh').read_text()
    bodies=[function(math,'__device__ __forceinline__ void _ModMultCore('),
            function(square,'__device__ __forceinline__ void qsb_square32(')]
    programs=[Program(extract_ptx(s)) for s in bodies]
    rng=random.Random(0xEA256)
    edges={0,1,2,C-1,C,C+1,P-2,P-1,P,P+1,P+2,B-2,B-1}
    for center in [1<<32,1<<64,1<<96,1<<128,1<<192,1<<224,1<<255,P-65537]:
        edges.update(x for x in range(center-2,center+3) if 0<=x<B)
    pairs=list(itertools.product(sorted(edges),repeat=2))
    pairs += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(2000)]
    counts={'add':0,'sub':0,'neg':0,'raw_multiply_PTX':0,'raw_square_PTX':0,
            'second_add_carries':0,'second_sub_borrows':0,'multiply_final_carries':0}
    for i,(a,b) in enumerate(pairs):
        add,ac=weak_add(a,b);sub,sc=weak_sub(a,b)
        assert 0<=add<B and norm(add)==(a+b)%P
        assert 0<=sub<B and norm(sub)==(a-b)%P
        # Third correction changes only the low64 word, with no lost carry/borrow.
        if ac:assert (add&((1<<64)-1))==add
        if sc:assert ((sub+C)&((1<<64)-1))>=C
        neg=weak_neg(a);assert 0<=neg<B and norm(neg)==(-a)%P
        counts['add']+=1;counts['sub']+=1;counts['neg']+=1
        counts['second_add_carries']+=ac;counts['second_sub_borrows']+=sc
        for op in (0,1):
            expected,carry=weak_product(a,a if op else b)
            raw=integer(programs[op].run(words(a)+([] if op else words(b))))
            assert raw==expected and 0<=raw<B and norm(raw)==a*(a if op else b)%P
            counts['raw_square_PTX' if op else 'raw_multiply_PTX']+=1
            counts['multiply_final_carries']+=carry
    toy=0
    for small_c in (3,5,9,13):
        small_b=256;small_p=small_b-small_c
        for a,b in itertools.product(range(small_b),repeat=2):
            add,_=weak_add(a,b,small_b,small_c);sub,_=weak_sub(a,b,small_b,small_c)
            assert 0<=add<small_b and add%small_p==(a+b)%small_p
            assert 0<=sub<small_b and sub%small_p==(a-b)%small_p
            toy+=1
    code=extract_ptx(bodies[0]);a=P-1;b=P-(1<<224)
    stale=code.replace('addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;')
    assert stale!=code and integer(Program(stale).run(words(a)+words(b)))%P!=a*b%P
    lost=code.replace('addc.u32 cf, 0, 0;','mov.u32 cf, 0;')
    assert lost!=code
    carry_witness=next((x,y) for x,y in pairs if weak_product(x,y)[1])
    assert integer(Program(lost).run(words(carry_witness[0])+words(carry_witness[1])))%P!=carry_witness[0]*carry_witness[1]%P
    defects={
      'old_add_weak_inputs':{'a':B-1,'b':B-1,'wrong':old_add(B-1,B-1),'expected':((B-1)*2)%P},
      'old_sub_weak_inputs':{'a':0,'b':B-1,'wrong':old_sub(0,B-1),'expected':(-(B-1))%P},
      'old_neg_weak_input':{'a':P+1,'wrong':old_neg(P+1),'expected':P-1},
      'omitted_second_add_carry':{'a':B-1,'b':B-1,'wrong':((B-2)+C)%B,'expected':2*C-2},
      'omitted_second_sub_borrow':{'a':0,'b':B-1,'wrong':(1-C)%B,'expected':P-C+1},
      'raw_zero_test':{'weak':P,'raw_nonzero':1,'field_nonzero':0},
      'raw_parity':{'weak':P+1,'raw_parity':(P+1)&1,'canonical_parity':1},
      'serialization':{'weak_x':P+1,'canonical_x':1},
      'canonical_inputs_produce_weak_output':{'a':P-1,'b':P-1,'weak_product':weak_product(P-1,P-1)[0],'canonical_product':1},
      'old_add_on_two_actual_weak_squares':{'canonical_square_input':P-65536,'weak_square':weak_product(P-65536,P-65536)[0],
          'wrong':old_add(weak_product(P-65536,P-65536)[0],weak_product(P-65536,P-65536)[0]),'expected':1<<33},
      'old_sub_on_actual_weak_square':{'canonical_square_input':P-1,'weak_square':weak_product(P-1,P-1)[0],
          'wrong':old_sub(0,weak_product(P-1,P-1)[0]),'expected':P-1},
      'full_carry_must_remain':{'a':carry_witness[0],'b':carry_witness[1]},
      'stale_cc_must_remain_rejected':{'a':a,'b':b},
    }
    for v in defects.values():
        if 'wrong' in v:assert v['wrong']%P!=v['expected']%P
    device=(SOURCE/'tests/gpu_epochs/compact_table_device.cuh').read_text()
    ordinary=function(device,'__device__ void qsb_ec192_PointAddXYZZ_shared_z_def(').split('  if (defer_y)')[0]
    op_counts={op:ordinary.count(op+'(') for op in ['_ModMult','_ModSqr','_ModAdd256','_ModSub256']}
    assert op_counts=={'_ModMult':7,'_ModSqr':2,'_ModAdd256':2,'_ModSub256':5},op_counts
    report={'status':'PASS_CONTRACT_MODELS_AND_ACTUAL_RAW_PTX_SEMANTICS',**identity,
       'checker_sha256':sha(Path(__file__)),'ptx_model_sha256':sha(ROOT/'research/ptx_field_model.py'),
       'counts':counts,'toy_exhaustive_pairs':toy,
       'source_PTX_ops':{'multiply':len(programs[0].ops),'square':len(programs[1].ops)},
       'witnesses':{name:{k:hex(v) if isinstance(v,int) else v for k,v in row.items()} for name,row in defects.items()},
       'ordinary_helper_source_counts':op_counts,
       'narrow_prototype_budget':{'iterations':13,'M_S_canonical_tails_removed':117,'end_normalizations_added':4,'weak_add_calls':13,'weak_sub_calls':65,'canonical_affine_Y_adds_retained':13,'persistent_field_words_before_after':[8,8]},
       'scope':'Exact unchanged source raw multiply/square PTX interpreted on CPU; independent integer weak add/sub/neg models and exhaustive toy-domain checks. No implemented CUDA weak add/sub, candidate mutation, GPU run, or full weak-circuit executable test.',
       'prior_distinction':'No new radix, no fifth limb or magnitude tracking, no raw product fusion. Remove only final canonical selection; all sparse-prime overflow folds remain.',
       'gpu_executed':False,'candidate_modified':False}
    (HERE/'contract-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':counts,'toy_pairs':toy,'budget':report['narrow_prototype_budget']}))
if __name__=='__main__':main()
