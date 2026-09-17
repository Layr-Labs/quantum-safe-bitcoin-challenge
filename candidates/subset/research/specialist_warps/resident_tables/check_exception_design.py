#!/usr/bin/env python3
"""Generic integer-domain proof checks; no new candidate, CUDA, or curve run."""
import hashlib,importlib.util,json,random,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));from preflight import source_identity
model_path=ROOT/'research/geometry_frontier/exception_domain.py'
spec=importlib.util.spec_from_file_location('prior_domain',model_path);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
SOURCE=HERE.parent/'two_six_ring/candidate';OLD=ROOT/'research/geometry_frontier/thirteen_candidate'
N=G.N;U=1<<256;DELTA=U-N

def digest(body):return hashlib.sha256(body.encode()).hexdigest()

def assess(widths):
    n=len(widths);shifts=[sum(widths[:i]) for i in range(n)];order=[n-2,n-1]+list(range(n-2));B=shifts[n-2];H=shifts[n-3];w=widths[n-3]
    assert sum(widths)==256 and DELTA.bit_length()==129 and DELTA<(1<<B)-(1<<H)
    seed_bound=U-(1<<B);assert seed_bound<N
    bounds=[]
    for c in range(n-3):
        S=shifts[c]+widths[c];lo=(1<<B)-(1<<S)+1;hi=U-(1<<B)+(1<<S)-1
        assert 0<lo<=hi<N
        bounds.append({'window':c,'end_bit':S,'min_A_plus_or_minus_Q':str(lo),'max_A_plus_or_minus_Q':str(hi),'margin_below_n':str(N-hi)})
    D=((1<<w)-1)<<H;eq=N-2*D
    assert 0<eq<N and eq&1 and D>DELTA and N-D>DELTA
    assert (N>>H)==(1<<(256-H))-1
    equal_domain=[]
    for d in range(-(1<<w)+1,0,2):
        M=N+2*(d<<H)
        extracted=(((M>>H)&((1<<(w+1))-1))|1)-(1<<w)
        if 1<=M<=N and extracted==d:equal_domain.append(M)
    assert equal_domain==[eq]
    expected_exception={0,N,D,N-D}
    cases=set(expected_exception)|{1,2,N-2,N-1,N+1,N+2,U-1}
    for k in [0,N,N//2,1<<255,U-1]+[1<<i for i in range(256)]:
        cases.update(k+x for x in range(-3,4) if 0<=k+x<U)
    inv2=pow(2,-1,N)
    for M in [1,3,N,eq]+[((q<<s)+d) for s in shifts for q in [0,1,(1<<(w-1))-1,1<<(w-1),(1<<w)-1] for d in [-3,-1,1,3]]:
        if 0<M<=N and M&1:
            k=M*inv2%N;cases.update([k,(-k)%N]);
            if k+N<U:cases.add(k+N)
    rng=random.Random(20260917+n);cases.update(rng.getrandbits(256) for _ in range(20000))
    counts={'scalars':0,'digits':0,'seed_sum_difference':0,'intermediate_sum_difference':0,'equal_final':0,'opposite_final':0,'wrong_top_last_detected':0,'wrong_sign_detected':0,'wrong_pos_detected':0}
    witnesses=[]
    for k in sorted(cases):
        m,sign=G.setup(k);M=G.integer(m);r=2*(k%N)%N
        assert (M,sign)==((r,1) if r&1 else (N-r,-1))
        state=m[:];digits=[]
        for c,bit in enumerate(widths):
            if c<n-1:state,peel=G.step(state,sign,bit)
            else:peel=sign*state[0]
            d,idx,neg=G.direct(m,sign,shifts[c]+1,bit,c==n-1)
            integer_digit=sign*(((M>>shifts[c])|1) if c==n-1 else ((((M>>shifts[c])&((1<<(bit+1))-1))|1)-(1<<bit)))
            assert d==peel==integer_digit and 0<=idx<1<<(bit-1) and (bool(neg)==(d<0))
            digits.append(d);counts['digits']+=1
        terms=[d<<s for d,s in zip(digits,shifts)];assert sum(terms)==sign*M
        A=terms[n-2];Q=terms[n-1]
        for v in [A+Q,A-Q]:
            assert (v>>B)&1 and v%(1<<B)==0 and 0<abs(v)<=seed_bound<N and v%N
            counts['seed_sum_difference']+=1
        A+=Q;assert A==sign*(((M>>B)|1)<<B)
        for c in range(n-3):
            Q=terms[c]
            for v in [A+Q,A-Q]:assert 0<sign*v<N and v%N;counts['intermediate_sum_difference']+=1
            A+=Q
        Q=terms[n-3]
        assert 0<sign*(A+Q)<U and 0<sign*(A-Q)<U
        equal=(A-Q)%N==0;opposite=(A+Q)%N==0
        assert equal==(k in {D,N-D}) and opposite==(k in {0,N})
        counts['equal_final']+=equal;counts['opposite_final']+=opposite
        if equal or opposite:witnesses.append({'raw_scalar':hex(k),'odd_M':hex(M),'sign':sign,'kind':'equal' if equal else 'opposite','A':str(A),'last_Q':str(Q),'last_digit':digits[n-3]})
        counts['wrong_top_last_detected']+=G.direct(m,sign,shifts[-1]+1,widths[-1],False)[0]!=digits[-1]
        counts['wrong_sign_detected']+=G.direct(m,-sign,shifts[0]+1,widths[0],False)[0]!=digits[0]
        counts['wrong_pos_detected']+=G.direct(m,sign,shifts[0],widths[0],False)[0]!=digits[0]
        counts['scalars']+=1
    assert {int(x['raw_scalar'],16) for x in witnesses}==expected_exception
    assert all(counts[k]>0 for k in ['wrong_top_last_detected','wrong_sign_detected','wrong_pos_detected'])
    # Removing the guarded final add exposes two equality and two opposite cases.
    assert counts['equal_final']==2 and counts['opposite_final']==2
    return {'widths':widths,'shifts':shifts,'order':order,'seed_order_reversed_also_safe':True,'only_top_last_flag_window':n-1,'guarded_final_window':n-3,'intermediate_count':n-3,'table_bytes':64*sum(1<<(bit-1) for bit in widths),'B':B,'H':H,'final_width':w,'seed_abs_bound':str(seed_bound),'deficit':str(DELTA),'gap_margin':str((1<<B)-(1<<H)-DELTA),'intermediate_bounds':bounds,'exhaustive_negative_final_digits':1<<(w-1),'unique_equal_representative':hex(eq),'D':hex(D),'complete_exception_raw_scalars':[hex(x) for x in sorted(expected_exception)],'counts':counts,'witnesses':witnesses,'unguarded_final_negative_control_exceptions':4}

def main():
    identity=source_identity(SOURCE);assert identity['source_fingerprint']=='32411a6a248c7ab53beba0f025450d95f31a0f2fc3da98a549456a47f93162d7'
    helper_specs=[('tests/gpu_epochs/tree.cu','__device__ __forceinline__ void gt_recode_setup('),('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ uint32_t gt_field_bits_v('),('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ void gt_direct_digit('),('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ void qsb_asym_last_add(')]
    helpers={}
    for path,sig in helper_specs:
        body=G.extract((SOURCE/path).read_text(),sig);prior=G.extract((OLD/path).read_text(),sig)
        assert body==prior,(path,sig)
        helpers[sig]=digest(body)
    results=[assess([18]+[17]*14),assess([16]*16)]
    assert source_identity(SOURCE)==identity
    out={'status':'PASS_GENERALIZED_INTEGER_DOMAIN_PROOF_AND_MODELS','base_source':identity,'unchanged_actual_helper_hashes':helpers,'limb_translation_source':'research/geometry_frontier/exception_domain.py','limb_translation_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometries':results,'correctness_basis':'All-input recoder identities and strict integer inequalities, plus exhaustive final-digit equality equation; finite samples validate source-matched translations, not the full-domain theorem.','preconditions':['Nonzero valid order-n secp256k1 runtime base','Correct runtime table construction and field primitives','Exact proposed order, signed recoder and top-last flag','Retain source final guard and seed old-Y anchor convention'],'no_candidate_generated':True,'gpu_executed':False,'limits':'Integer/limb models, not extracted C++ execution for either future geometry. No new table builder, helper-chain, or startup-policy qualification.'}
    (HERE/'exception-design.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{'widths':r['widths'],'order':r['order'],'counts':r['counts'],'exceptions':r['complete_exception_raw_scalars']} for r in results],indent=2))
if __name__=='__main__':main()
