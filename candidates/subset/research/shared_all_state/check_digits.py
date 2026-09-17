#!/usr/bin/env python3
"""Apply existing full-uint256 recoder oracle to actual shared digit helpers."""
import hashlib,json,pathlib,sys
sys.dont_write_bytecode=True
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research/wide_windows')]
from audit_support import function
source=HERE/'candidate';header=(source/'tests/gpu_epochs/compact_table_device.cuh').read_text()
shared='\n'.join(function(header,s) for s in ['__device__ __forceinline__ uint32_t qsb_shared_field_bits(','__device__ __forceinline__ void qsb_shared_direct_digit('])
legacy=ROOT/'research/pr120_digits/check_digits.py';adapter=legacy.read_text()
marker="code+=r'''"
assert adapter.count(marker)==1
adapter=adapter.replace(marker,"code += "+repr(shared)+"\n"+marker)
marker=' uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);compact_recode_signed(k,ref);'
assert adapter.count(marker)==1
adapter=adapter.replace(marker,marker+'''
 static unsigned serial=0;unsigned tid=(serial++*73)&255u;
 uint64_t arena[4][1024];
 for(int j=0;j<4;j++)for(int lane=0;lane<1024;lane++)arena[j][lane]=0x916312be075cba98ULL;
 for(int j=0;j<4;j++)arena[j][tid]=M[j];
''')
# Replace actual C++ audit calls only. Existing source assertions still confirm
# donor helpers retained unchanged, but the tested arithmetic is shared helper.
pos=adapter.index("code+=r'''");end=adapter.index("\n'''",pos)
audit=adapter[pos:end].replace('gt_direct_digit(M,','qsb_shared_direct_digit(arena,tid,')
adapter=adapter[:pos]+audit+adapter[end:]
adapter=adapter.replace("'actual_chain_sha256':sha(chain)","'actual_chain_sha256':sha(function(header,'__device__ void compact_fixed_xyzz_shared('))")
adapter=adapter.replace("'validation_level':'Actual extracted donor helper, setup, unchanged reference peel and cold helper on CPU with UBSan; independent Python regular recoder'","'validation_level':'Actual extracted shared-arena digit helpers with varied owner tid, setup, reference peel and cold helper on CPU with UBSan; independent Python regular recoder'")
sys.argv=[str(legacy),'--source',str(source),'--report',str(HERE/'digits-results.json')]
exec(compile(adapter,str(legacy)+'[shared-digit-projection]','exec'),{'__name__':'__main__','__file__':str(legacy)})
p=HERE/'digits-results.json';r=json.loads(p.read_text());r['actual_shared_helpers_sha256']=hashlib.sha256(shared.encode()).hexdigest();r['adapter_sha256']=hashlib.sha256(adapter.encode()).hexdigest();p.write_text(json.dumps(r,indent=2)+'\n')
