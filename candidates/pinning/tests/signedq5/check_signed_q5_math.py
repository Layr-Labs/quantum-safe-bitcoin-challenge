#!/usr/bin/env python3
"""Bounded source-bound signed-Q recoding model; Python standard library only."""
from pathlib import Path
import argparse, base64, bisect, hashlib, itertools, json, random, re, time

if not __debug__:
    raise RuntimeError("Run without -O: this proof requires assertion checks")
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2],
                    help='production candidates/pinning directory (default assumes tests/signedq5 placement)')
parser.add_argument('--receipt',type=Path,help='optional JSON output; production sources are never written')
args=parser.parse_args()
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
binding=json.loads((D/'source-manifest.json').read_text())
root=args.source_root.resolve()
assert len(binding['candidateProduction39'])==39
for name,digest in binding['candidateProduction39'].items():
    assert '/' not in name and name not in ('.','..')
    p=root/name
    assert p.is_file() and not p.is_symlink(), name
    assert sha(p.read_bytes())==digest, ('production source hash mismatch',name)
candidate=(root/'pinning.cu').read_text()
patch=(D/'baseline-to-candidate.patch').read_bytes()
assert sha(patch)==binding['baselineToCandidatePatchSHA256']

def reverse_patch(candidate,patch):
    """Strict in-memory reversal; match every context and added byte."""
    src=candidate.splitlines(keepends=True)
    lines=patch.decode().splitlines(keepends=True)
    result=[];pos=0;i=0;hunks=0
    while i<len(lines):
        if not lines[i].startswith('@@ '):
            assert i<2 and lines[i].startswith(('--- ','+++ ')), lines[i]
            i+=1;continue
        m=re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n',lines[i])
        assert m, lines[i]
        oldcount=int(m[2] or 1);newstart=int(m[3])-1;newcount=int(m[4] or 1)
        before=[];after=[];i+=1
        while i<len(lines) and not lines[i].startswith('@@ '):
            line=lines[i];assert line[:1] in (' ','+','-'),line
            if line[0] in ' -':before.append(line[1:])
            if line[0] in ' +':after.append(line[1:])
            i+=1
        assert len(before)==oldcount and len(after)==newcount
        assert newstart>=pos and src[newstart:newstart+newcount]==after
        result.extend(src[pos:newstart]);result.extend(before)
        pos=newstart+newcount;hunks+=1
    assert hunks==binding['patchHunks']
    result.extend(src[pos:])
    return ''.join(result)

text=reverse_patch(candidate,patch)
assert sha(text.encode())==binding['baselinePinningSHA256']
assert '#define QSB_SIGNED_Q5_ENDPOINT 1' in candidate
assert '#define QSB_QMIX5_SEL() true' in candidate
assert 'if(!q5) codes[(size_t)2*QSB_TREE_N+threadIdx.x]=q9_bigtbl_code_z(w,top,m32,2);' in candidate
glv=(root/'GLVScalar.cuh').read_text()
math=(root/'GPUMath.h').read_text()
tree=(root/'cofactor_checkpoint.h').read_text()
header=(root/'qsb_carrier_sm89.h').read_text()
array=header.split('static const char *const qsb_carrier_b64[] = {',1)[1].split('};',1)[0]
chunks=re.findall(r'"([A-Za-z0-9+/=]+)"',array)
image=base64.b64decode(''.join(chunks),validate=True)
assert len(image)==binding['candidateNativeBytes']
assert sha(image)==binding['candidateNativeSHA256']
for required in (
    '#define QSB_GLV11 1', '#define QSB_QGLV5 0', '#define QSB_PDEC_Z 1',
    '#define QSB_PMIX12 65536', '#define QSB_PMIX12_WARP 0', '#define QSB_QMIX5 8',
    '#define QSB_GLV_SEED_REG 1', '#define QSB_CHAIN_PEEL 1', '#define QSB_PHI_HOIST 1'):
    assert required in text, required
for required in ('#define QSB_GT_TOP_CENTER 170559769u',
                 'return c==0?0u:c==1?18u:c==2?37u:c==3?55u:c==4?73u:100u;',
                 'if(c>=6) return c==6?18u:45u;',
                 'if(c>=6) return c==6?153175181u:220284045u;',
                 '#define QSB_GLV_NZ_CUT 1', '#define QSB_DECODE_CUT 3'):
    assert required in glv, required
assert 'BN_set_word(bias,QSB_GT_TOP_CENTER+1u); BN_lshift(bias,bias,QSB_GT_TOP_SHIFT-1u); BN_sub_word(bias,1u<<17);' in text
assert 'BN_mul_word(k,(BN_ULONG)(2*index+1));' in text
assert '#define QSB_MUL_FOLD8_CUT 1' in math and '#define QSB_SQR_FOLD8_CUT 1' in math
assert '#define QSB_GF_P(i) (i)' in tree
assert 'qsb_tv_st(T,16u*QSB_GF_P(tid),value);\n    qsb_tv_st(T,16u*ucol,U);\n    __syncthreads();' in tree
assert '#define QSB_AB_SW_OFF 6144' in text
# All lanes initialize the paired leaf's first limb pair before tree reads. It
# covers every byte of the omitted Qslot2 plane, after the chain handoff barrier.
slot2_bytes=set(range(2*128*4,3*128*4))
first_tree_store_bytes={16*tid+i for tid in range(128) for i in range(16)}
assert slot2_bytes <= first_tree_store_bytes

U32 = (1<<32)-1; U128 = (1<<128)-1; SIGN = 1<<31
CEN = 170559769
K = (CEN+1)*(1<<99)-(1<<17)
BOUND = int('a2a8918ca85bafe22016d0b917e4dd77',16)
SH = [0,18,37,55,73,100,18,45]
BITS = [18,19,18,18,27,None,27,28]
OFF = [0,262144,524288,655360,786432,67895296,153175181,220284045]
ENT = [262144,262144,131072,131072,67108864,85279885,67108864,134217728]
TOTAL = 354501773
assert BOUND>>100 == CEN-1 and OFF[-1]+ENT[-1] == TOTAL
Q6 = [0,1,2,3,4,5]; Q5 = [0,6,7,4,5]

def lop(a,b,c,lut):
    out=0
    for i in range(8):
        if lut>>i&1:
            out |= (a if i&4 else a^U32) & (b if i&2 else b^U32) & (c if i&1 else c^U32)
    return out&U32

# All eight boolean inputs independently establish each literal PTX truth table.
lop_checks=0
for a,b,c in itertools.product((0,U32),repeat=3):
    for lut,expected in ((0x28,(a^b)&c),(0xF8,a|(b&c)),
                         (0xC3,a^(b^U32)),(0xD2,a^((b^U32)&c))):
        assert lop(a,b,c,lut) == expected&U32
        lop_checks+=1

def mag_code(mag, sign, seg):
    """Independent field/magnitude model, without signed-W windows."""
    f=mag>>SH[seg]
    if seg==0:
        index=f&((1<<18)-1); neg=0
    elif seg==5:
        digit=2*f-CEN
        neg=int(digit<0); index=(abs(digit)-1)//2
    else:
        bits=BITS[seg]; f&=(1<<bits)-1
        digit=2*f-(1<<bits)+1
        neg=int(digit<0); index=(abs(digit)-1)//2
    return OFF[seg]+index | ((neg^sign)<<31)

def scalar(code):
    """Decode actual physical table segment and its gt_table_scalar value."""
    record=code&0x7fffffff
    assert 0<=record<TOTAL, record
    seg=bisect.bisect_right(OFF,record)-1
    index=record-OFF[seg]
    assert index<ENT[seg]
    value=K+index if seg==0 else (2*index+1)*(1<<(SH[seg]-1))
    return -value if code&SIGN else value

def words(w):
    return [(w>>(32*i))&U32 for i in range(4)]

def radix(ws, seg):
    """Literal qsb_qmix5_radix / q11_radix_code_z 32-bit PTX arithmetic."""
    bits=BITS[seg]; r=SH[seg]+bits-32; q,rs=divmod(r,32)
    assert 0<=q<3 and 0<=rs<32
    u=(((ws[q+1]<<32)|ws[q])>>rs)&U32
    xs=U32 if u&SIGN else 0
    v=(((u^(xs^U32))>>(32-bits))+OFF[seg]-(1<<(bits-1)))&U32
    code=(v^((u^U32)&SIGN))&U32
    return v,xs,code

def z_code(w,sign,seg):
    ws=words(w); m=U32 if sign else 0
    if seg==0:
        return ((ws[0]^m)&((1<<18)-1)) | (m&SIGN)
    if seg==5:
        h=(((ws[3]^m)>>4)-(CEN+1)//2)&U32
        xs=U32 if h&SIGN else 0
        return (OFF[seg]+(h^xs)+((h^m)&SIGN))&U32
    return radix(ws,seg)[2]

def qmix(w, sign, q5):
    """Literal seed/mask + slot writes; slot2 is a sentinel when guarded out."""
    ws=words(w); m=U32 if sign else 0
    seed0=(ws[0]^m)&((1<<18)-1)
    msk0=m
    seed1,msk1,_=radix(ws,6 if q5 else 1)
    # Seed0 uses NEG=false; seed1 uses NEG=true and receives the inverted mask.
    code0=seed0 | ((msk0>>31)<<31)
    code1=seed1 | ((((msk1^U32)>>31)&1)<<31)
    slots={2:None if q5 else z_code(w,sign,2),
           3:radix(ws,7 if q5 else 3)[2],4:z_code(w,sign,4),5:z_code(w,sign,5)}
    codes=[code0,code1]+[slots[i] for i in ([3,4,5] if q5 else [2,3,4,5])]
    assert None not in codes
    return codes,slots

def geometric_exceptions(codes):
    # Every scalar in the Q prefix has abs(value)<<n, so integer +/- equality
    # is also equality/opposition modulo the order n for these prefix tests.
    acc=scalar(codes[0]); exceptions=[]
    for i,code in enumerate(codes[1:],1):
        nxt=scalar(code)
        if acc==nxt: exceptions.append(('double','last_Q' if i==len(codes)-1 else i))
        if acc==-nxt: exceptions.append(('infinity','last_Q' if i==len(codes)-1 else i))
        acc+=nxt
    return exceptions

t0=time.monotonic()
rng=random.Random(0x514d495835)
mags={0,1,2,BOUND,BOUND-1,(CEN-1)<<100,((CEN-1)<<100)-1,((CEN-1)<<100)+1}
for bit in range(128):
    for d in (-1,0,1):
        m=(1<<bit)+d
        if 0<=m<=BOUND: mags.add(m)
for shift in (18,37,45,55,73,100):
    for field in (0,1,2,127,128,255,256,511,512,1023,1024,(1<<17)-1,1<<17,
                  (1<<18)-1,1<<18,(1<<26)-1,1<<26,(1<<27)-1,1<<27):
        for low in (0,1,(1<<shift)-1):
            m=(field<<shift)|low
            if m<=BOUND: mags.add(m)
# Cross the repartition boundary at45: low8 old-segment2 bits and its high10 bits.
for low8 in range(256):
    for high10 in (0,1,511,512,1023):
        for old3 in (0,1,(1<<17)-1,1<<17,(1<<18)-1):
            m=(low8<<37)|(high10<<45)|(old3<<55)
            assert m<=BOUND
            mags.add(m)
for _ in range(8192): mags.add(rng.randrange(BOUND+1))
mags=sorted(mags)
rows=code_checks=midpoint_checks=0
range_min=[TOTAL]*8; range_max=[-1]*8
for mag in mags:
    for sign in (0,1):
        w=mag^(U128 if sign else 0)
        out={}
        for five,segs in ((False,Q6),(True,Q5)):
            actual,slots=qmix(w,sign,five)
            expected=[mag_code(mag,sign,s) for s in segs]
            assert actual==expected,(mag,sign,five,actual,expected)
            for code,seg in zip(actual,segs):
                record=code&0x7fffffff
                assert OFF[seg]<=record<OFF[seg]+ENT[seg],(mag,sign,seg,record)
                range_min[seg]=min(range_min[seg],record-OFF[seg])
                range_max[seg]=max(range_max[seg],record-OFF[seg])
                code_checks+=1
            value=sum(map(scalar,actual))
            assert value==(-mag if sign else mag),(mag,sign,five,value)
            out[five]=actual
            rows+=1
        # Same partial point before common segment4 and again before common top5.
        assert sum(map(scalar,out[False][:-2]))==sum(map(scalar,out[True][:-2]))
        assert sum(map(scalar,out[False][:-1]))==sum(map(scalar,out[True][:-1]))
        assert geometric_exceptions(out[False])==geometric_exceptions(out[True])
        midpoint_checks+=2

# Endpoint equality for arbitrary128-bit synthetic residuals, including shared top
# records aliasing another physical segment outside the true magnitude bound.
synthetic_checks=0; borrow_checks=0
for mag in [0,U128,U128-1,CEN<<100,(CEN<<100)-1,(CEN<<100)+1]+[rng.getrandbits(128) for _ in range(2048)]:
    for sign in (0,1):
        w=mag^(U128 if sign else 0)
        six,_=qmix(w,sign,False); five,_=qmix(w,sign,True)
        assert sum(map(scalar,six))==sum(map(scalar,five))
        synthetic_checks+=1
for amount in (1,2,3,(1<<31)-1,1<<31):
    mag=amount<<32
    # Actual DECODE_CUT=1 negative borrow skip creates Wexact+2^32 mod2^128.
    w=((mag^U128)+(1<<32))&U128
    six,_=qmix(w,1,False); five,_=qmix(w,1,True)
    assert sum(map(scalar,six))==sum(map(scalar,five))
    borrow_checks+=1

def schedule(pmag,psign,qmag,qsign,q5,p6,nonzero):
    """Slot-read and phi model of the literal seed/peeled one-add chain."""
    p_segs=Q6 if p6 else Q5
    p_codes=[mag_code(pmag,psign,s) for s in p_segs]
    q_codes,q_slots=qmix(qmag^(U128 if qsign else 0),qsign,q5)
    slots={**q_slots,**{6+i:c for i,c in enumerate(p_codes)}}
    first=(1 if q5 else 0) if nonzero&1 else 6
    last=11+int(p6)
    if nonzero&1:
        seeds=q_codes[:2]; reads=[]
    else:
        reads=[6,6+(nonzero>>1)]; seeds=[slots[i] for i in reads]
    if nonzero==0:
        assert seeds[0]==seeds[1] # literal seed X1==X2 -> U=V=0
        return None,reads,0
    point=[sum(map(scalar,seeds)),0]; phi=0
    # Preload first+2 then PIPE loads the next term, final trip consumes it.
    for term in range(first+2,last):
        if term==6:
            point=[0,point[0]]; phi+=1
        reads.append(term)
        assert slots[term] is not None,(first,term,q5,p6,nonzero)
        point[0]+=scalar(slots[term])
    return tuple(point),reads,phi

schedule_checks=0; block_checks=0
for block in range(1024):
    a5=(block&7)==0; p6=(block&65535)==0
    assert p6==(block==0)
    for nonzero in (0,1,2,3):
        for psign,qsign in itertools.product((0,1),repeat=2):
            pmag=0 if not(nonzero&2) else 0x90123456789abcdef
            qmag=0 if not(nonzero&1) else 0xfedcba987654321
            pa,reads_a,phi_a=schedule(pmag,psign,qmag,qsign,a5,p6,nonzero)
            pc,reads_c,phi_c=schedule(pmag,psign,qmag,qsign,True,p6,nonzero)
            assert pa==pc and phi_a==phi_c
            assert 2 not in reads_c
            if nonzero:
                expect=(-pmag if psign else pmag,-qmag if qsign else qmag)
                assert pc==expect
                assert phi_c==int(bool(nonzero&1))
            schedule_checks+=1
    block_checks+=1
# Selector semantics also at the next rare-P boundary65536, not just launchCTA0.
for block in (65535,65536,65537):
    for signs in itertools.product((0,1),repeat=2):
        ps,qs=signs; p6=(block&65535)==0
        assert schedule(BOUND,ps,BOUND,qs,(block&7)==0,p6,3)[0] == schedule(BOUND,ps,BOUND,qs,True,p6,3)[0]
        schedule_checks+=1

bias6=sum(((1<<BITS[s])-1)*(1<<(SH[s]-1)) for s in (1,2,3,4))
bias5=sum(((1<<BITS[s])-1)*(1<<(SH[s]-1)) for s in (6,7,4))
assert bias6==bias5==(1<<99)-(1<<17)
exceptions={str(m):{'Q6':geometric_exceptions(qmix(m,0,False)[0]),
                    'Q5':geometric_exceptions(qmix(m,0,True)[0])}
            for m in (0,CEN<<100)}
receipt={
    'schema':'qsb-signed-q5-public-bounded-reference-v1',
    'status':'PASS_RECODING_CODES_AND_SLOT_PHI_MODEL_NOT_NATIVE_FIELD_EQUIVALENCE',
    'scriptSHA256':sha(Path(__file__).read_bytes()),
    'sourceManifestSHA256':sha((D/'source-manifest.json').read_bytes()),
    'candidateProduction39ByteBound':True,
    'baselinePinningReconstructedAndByteBound':True,
    'baselineCommit':binding['baselineCommit'],
    'literalSources':{n:binding['candidateProduction39'][n] for n in ('GLVScalar.cuh','GPUMath.h','cofactor_checkpoint.h')},
    'baselinePinningSHA256':binding['baselinePinningSHA256'],
    'candidatePinningSHA256':binding['candidateProduction39']['pinning.cu'],
    'candidateHeaderDecodedNativeSHA256':sha(image),
    'candidateHeaderDecodedNativeBytes':len(image),
    'tableBiasInteger':str(K),'ordinaryBiasBoth':str(bias6),
    'boundHex':hex(BOUND),'trueBoundTopFieldMax':CEN-1,'declaredSegment5TopFieldRangeMax':CEN,
    'lop3BooleanTruthCases':lop_checks,'uniqueBoundedMagnitudes':len(mags),
    'bothSignAndRouteRows':rows,'codeSeedMaskChecks':code_checks,
    'commonMidPrefixSumChecks':midpoint_checks,
    'syntheticFull128BitEndpointChecks':synthetic_checks,
    'activeDecodeCutBorrowEndpointChecks':borrow_checks,
    'all1024BlockSelectorChecks':block_checks,
    'slotPhiScheduleChecksIncludingRare65536CTA':schedule_checks,
    'omittedSlot2PlaneBytesOverwrittenByInitialTreeLeafStoresBeforeReads':len(slot2_bytes),
    'ASICScheduleByteOffsetUnchangedAndAboveDigitSlots':6144,
    'observedLocalRecordIndexRanges':[{'segment':s,'min':range_min[s],'max':range_max[s],'entries':ENT[s]} for s in range(8)],
    'sharedQExceptionalGeometry':exceptions,
    'syntheticTopDoublingMagnitudeAboveExactResidualBound':(CEN<<100)>BOUND,
    'activeInheritedSwitches':{'QSB_GLV_NZ_CUT':1,'QSB_DECODE_CUT':3,
                              'QSB_MUL_FOLD8_CUT':1,'QSB_SQR_FOLD8_CUT':1,
                              'QSB_HIGH15_NOFB':1,'QSB_ZSPLIT_NOPRE':1,'QSB_YOFF_Y1_CUT':1},
    'limits':['finite Python reference plus algebra; no compiled PTX or native execution',
              'same decoded W is the comparison input; upstream coefficient/borrow policies unchanged',
              'default NZ_CUT forces nonzero=3 even for Q0; fallback schedule checks are conditional source semantics',
              'ideal group identity does not prove bitwise native field behavior with existing rare-carry cuts',
              'projective X/Y/U/V may differ; qualify affine/recovered output and usable status, not raw equality'],
    'elapsedCPUSeconds':time.monotonic()-t0,
    'compileCount':0,'GPUCount':0,'productionFilesMutated':False,
}
if args.receipt:
    destination=args.receipt.resolve()
    assert destination!=root and root not in destination.parents, "Receipt must be outside production source directory"
    destination.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
