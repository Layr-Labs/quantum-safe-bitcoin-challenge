#!/usr/bin/env python3
"""Execute PR137's actual host packer and count source-derived address groups.

No CUDA/cache simulation or speed claim. Preserve exact unpromoted provenance.
"""
import collections
import ctypes as C
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from check_candidate import function

COMMIT='103a6adc15391e07aac210a2653f8dfd7eb4d4c0'
PATH='candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh'


def key(w,second):
    kept=[x for x in range(137,150) if x not in w]
    return tuple(reversed(kept[-5:])) if second else tuple(kept[:6])


def counts(windows):
    # qsb_prepare_window_schedule assigns slots by first encounter in lane order.
    first_slots={};second_slots={};fs=[];ss=[]
    for w in windows:
        f,s=key(w,False),key(w,True)
        fs.append(first_slots.setdefault(f,len(first_slots)))
        ss.append(second_slots.setdefault(s,len(second_slots)))
    groups=[];sectors=[];lines=[];banks=[]
    for start in range(0,256,32):
        s=set(ss[start:start+32]);f=set(fs[start:start+32])
        groups.append(len(s));sectors.append(len({x//8 for x in s}))
        lines.append(len({x//32 for x in s}))
        banks.append(max(collections.Counter(x%32 for x in f).values()))
    return dict(first_classes=len(first_slots),second_classes=len(second_slots),
                warp_second_classes=groups,second_class_group_sum=sum(groups),
                warp_second_32byte_sector_groups=sectors,sector_group_sum=sum(sectors),
                warp_second_128byte_line_groups=lines,line_group_sum=sum(lines),
                first_state_worst_bank_multiplicity_per_warp=banks,
                first_state_bank_multiplicity_sum=sum(banks))


def main():
    source=subprocess.check_output(['git','show',COMMIT+':'+PATH],text=True)
    symbols=['static uint32_t qsb_window_second_key(',
             'static int qsb_pack_second_classes(']
    bodies=[function(source,s) for s in symbols]
    (Path(__file__).parent/'public-packer.cxx').write_text(
        '// GPL-derived PR137 helper; IvanLudvig, unpromoted source '+COMMIT+'\n'
        +'#include <stdint.h>\n#include <cstring>\n'+'\n'.join(bodies))
    windows=[w for w in itertools.combinations(range(137,150),3)
             if not(w[0]>=138 and w[2]<=144 and not(w[0]==138 and w[1]==139))]
    windows.sort(key=lambda w:(key(w,True),key(w,False)))
    assert len(windows)==256
    with tempfile.TemporaryDirectory(prefix='qsb-warp-packer-') as tmp:
        p=Path(tmp);cpp=p/'check.cpp';lib=p/'check.so'
        cpp.write_text('#include <stdint.h>\n#include <cstring>\n'+'\n'.join(bodies)+
                       '\nextern "C" int pack(uint8_t*x){return qsb_pack_second_classes((uint8_t(*)[3])x);}\n')
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
        dll=C.CDLL(str(lib));dll.pack.argtypes=[C.POINTER(C.c_uint8)]
        raw=(C.c_uint8*768)(*[v for w in windows for v in w]);assert dll.pack(raw)==0
        packed=[tuple(raw[3*i:3*i+3]) for i in range(256)]
    assert sorted(packed)==sorted(windows)
    before,after=counts(windows),counts(packed)
    assert before['second_class_group_sum']==62 and after['second_class_group_sum']==56
    assert before['first_classes']==after['first_classes']==54
    assert before['second_classes']==after['second_classes']==56
    result=dict(status='PASS',source_commit=COMMIT,source_url='https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/137',
                coauthor_if_used='IvanLudvig',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                actual_helper_sha256={symbol:hashlib.sha256(body.encode()).hexdigest() for symbol,body in zip(symbols,bodies)},
                exact_candidate_multiset_preserved=True,before=before,after=after,
                decision='Supporting hypothesis only; class-group savings do not directly count physical fetches or predict throughput.',
                assumptions=['32-byte sector and128-byte line groups are source address group counts, not cache misses or DRAM transactions.',
                             'Static shared-bank model uses32banks with4-byte words and broadcasts identicaladdresses; not a runtime stall measure.',
                             'Assumes aligned QSB_WINDOW_SECOND[64][256] and first_states[8][256] declarations from selected source.',
                             'No paired-SHA helper or host-drain changes imported.'],gpu_executed=False)
    (Path(__file__).parent/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'before':before,'after':after},indent=2))


if __name__=='__main__':main()
