#!/usr/bin/env python3
"""Create a temporary source closure with an exact split-word MAC seed."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
BASELINE_COMMIT='7e95c40c99e57bded233ce57c7f453fbde9fd21c'
BASELINE_SHA256='d8893d08e76acd5c9c00531e89783c309e1dab78c3cce6509e4c667e36ac0335'

def baseline():
    """Load the immutable pre-experiment header, never patch selected production twice."""
    data=subprocess.run(['rtk','proxy','git','show',
        BASELINE_COMMIT+':candidates/pinning/negative_y_mac.cuh'],
        cwd=SOURCE.parents[1],check=True,capture_output=True,text=True).stdout
    assert hashlib.sha256(data.encode()).hexdigest()==BASELINE_SHA256
    return data

def variant(data,narrow=False):
    marker='__device__ __forceinline__ void qsb_muladd_seed('
    assert data.count(marker)==1
    assert 'QSB_MAC_HALF_SEED' not in data, 'Expected the pinned unpatched baseline'
    flags="""#ifndef QSB_MAC_HALF_SEED
#define QSB_MAC_HALF_SEED 1
#endif
#if QSB_MAC_HALF_SEED != 0 && QSB_MAC_HALF_SEED != 1
#error QSB_MAC_HALF_SEED must be 0 or 1
#endif
// Seed each 32-bit product column separately. For B=2^32,
// (B-1)^2+(B-1)=B^2-B, so these eight MADs cannot overflow.
// The existing full product and reduction schedules are otherwise unchanged.
"""
    data=data.replace(marker,flags+marker)
    start=data.index('  "mul.wide.u32 e0, a0, b0;')
    stop=data.index('  "mul.wide.u32 t, a1, b1;',start)
    old=data[start:stop]
    lines=['.reg .u32 c0,c1,c2,c3,c4,c5,c6,c7;']
    if narrow:lines.append('.reg .u32 seeded_lo,seeded_hi;')
    for i in range(4):lines.append(f'mov.b64 {{c{2*i},c{2*i+1}}}, %{12+i};')
    for i in range(8):
        dest=('e' if i%2==0 else 'o')+str(i//2)
        if narrow:
            lines += [f'mad.lo.cc.u32 seeded_lo, a0, b{i}, c{i};',
                      f'madc.hi.u32 seeded_hi, a0, b{i}, 0;',
                      f'mov.b64 {dest}, {{seeded_lo,seeded_hi}};']
        else:
            lines += [f'cvt.u64.u32 bias, c{i};',f'mad.wide.u32 {dest}, a0, b{i}, bias;']
    fresh=''.join('  '+json.dumps(line+'\n')+'\n' for line in lines)
    data=data[:start]+'#if QSB_MAC_HALF_SEED\n'+fresh+'#else\n'+old+'#endif\n'+data[stop:]
    old='  "addc.u64 e4, t, bias_carry;\\n"'
    assert data.count(old)==1
    data=data.replace(old,'#if QSB_MAC_HALF_SEED\n  "addc.u64 e4, t, 0;\\n"\n#else\n'+old+'\n#endif')
    return data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--narrow',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    seen=set()
    def copy(name):
        if name in seen:return
        seen.add(name);raw=(SOURCE/name).read_bytes()
        for inc in re.findall(rb'^\s*#include\s+"([^"\n]+)"',raw,re.M):
            if (SOURCE/inc.decode()).is_file():copy(inc.decode())
        if name=='negative_y_mac.cuh':raw=variant(baseline(),args.narrow).encode()
        (args.out/name).write_bytes(raw)
    copy('pinning.cu')
    print(json.dumps({'out':str(args.out),'files':len(seen),
          'header_sha256':hashlib.sha256((args.out/'negative_y_mac.cuh').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
