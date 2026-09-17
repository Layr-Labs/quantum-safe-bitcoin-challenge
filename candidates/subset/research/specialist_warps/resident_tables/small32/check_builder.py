#!/usr/bin/env python3
"""Small32 actual-source builder checks using preserved OpenSSL/cooperative adaptors.
No source mutation, GPU execution or native CUDA build. CPU projections omit
unused search-subgroup helpers, while preserving actual startup checkpoint bodies.
"""
import argparse,hashlib,json,re,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from preflight import source_identity

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project_run(original,replacements,argv):
    code=original.read_text()
    for old,new in replacements:
        assert old in code,old
        code=code.replace(old,new)
    oldargv=sys.argv
    try:
        sys.argv=[str(original)]+argv
        exec(compile(code,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
    finally:sys.argv=oldargv
    return hashlib.sha256(code.encode()).hexdigest()

def descriptor_check(src,out):
    from check_candidate import function
    h=src/'tests/gpu_epochs'
    decl=(h/'compact_table_device.cuh').read_text().split('// Direct regular-digit')[0]
    decl=re.sub(r'^#include "[^"\n]+"\n','',decl,flags=re.M)
    decoder=function((h/'compact_table_kernels.cuh').read_text(),'__host__ __device__ __forceinline__ void mixed_decode_entry(')
    code='#include <cstdint>\n#include <cstdio>\n'+(h/'compact_geometry.cuh').read_text()+'\n'+decl+'\n'+decoder+r'''
int main(){
 if(COMPACT_CHUNKS!=16||COMPACT_TOTAL_ENTRIES!=(1u<<19)||COMPACT_LO!=256||COMPACT_HI!=256)return 2;
 for(int c=0;c<16;c++)if(mixed_bits(c)!=16||compact_entries(c)!=(1u<<15)||compact_offset(c)!=(unsigned(c)<<15)||compact_shift(c)!=16*c)return 2;
 if(compact_offset(16)!=(1u<<19))return 2;
 for(uint32_t t=0;t<(1u<<19);t++){
  int ch;uint32_t hi,lo;mixed_decode_entry(t,&ch,&hi,&lo);
  unsigned want_ch=t/32768u,m=2*(t%32768u)+1;
  if(ch!=int(want_ch)||hi!=m/256u||lo!=m%256u){fprintf(stderr,"DECODE_MISMATCH t=%u expected_hi=%u got_hi=%u\n",t,m/256u,hi);return 3;}
 }
 puts("DESCRIPTOR_AND_ALL524288_DECODES_PASS");
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-small32-descriptor-') as tmp:
        cpp=Path(tmp)/'descriptor.cpp';exe=Path(tmp)/'descriptor'
        outputs=[]
        for mutated in [False,True]:
            projection=code.replace('*hi=m>>8;','*hi=m>>9;') if mutated else code
            cpp.write_text(projection)
            subprocess.run(['c++','-std=c++17','-O2','-Wno-pragma-once-outside-header',str(cpp),'-o',str(exe)],check=True)
            result=subprocess.run([str(exe)],capture_output=True,text=True)
            if not mutated:assert result.returncode==0,result.stderr
            else:assert result.returncode==3 and 'DECODE_MISMATCH t=128 expected_hi=1 got_hi=0' in result.stderr,result
            outputs.append({'mutation':'hi shift8 to9' if mutated else None,'projection_sha256':hashlib.sha256(projection.encode()).hexdigest(),'compile_pass':True,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    d={'status':'PASS','scope':'Actual geometry/declaration/decoder source on CPU; independent exact16bit geometry oracle.','descriptor_windows':16,'decode_cases':524288,'boundary_sentinel_checked':True,'runs':outputs}
    (out/'descriptor.json').write_text(json.dumps(d,indent=2)+'\n')
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=HERE/'candidate');ap.add_argument('--output',type=Path,default=HERE/'builder-results');ap.add_argument('--skip-pipeline',action='store_true');a=ap.parse_args()
    src=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    start=source_identity(src);h=src/'tests/gpu_epochs';host=(h/'compact_table_host.cuh').read_text();kernels=(h/'compact_table_kernels.cuh').read_text()
    assert '#define COMPACT_LO 256' in (h/'compact_table_device.cuh').read_text()
    assert '#define COMPACT_HI 256' in (h/'compact_table_device.cuh').read_text()
    assert '*ch=(int)(t>>15);' in kernels and '*hi=m>>8;*lo=m&255u;' in kernels
    assert '4095' not in host and '8192' not in kernels
    assert 'COMPACT_LO/2-1,COMPACT_LO/2,COMPACT_LO/2+1' in host
    assert 'high_count<=COMPACT_HI' in host and 'BN_bn2binpad(x,xb,32)!=32' in host
    descriptor=descriptor_check(src,out)
    wide=ROOT/'research/wide_windows/check_builder.py'
    sys.path.insert(0,str(wide.parent))
    widths=','.join(['16']*16)
    wide_projection=project_run(wide,[('maps=entries+[rng.randrange(total) for _ in range(100000)]','maps=list(range(total))')],['--source',str(src),'--compact','--widths',widths,'--low-bits','8','--report',str(out/'ladder-affine.json')])
    pipeline_projection=None
    if not a.skip_pipeline:
        pipeline=ROOT/'research/pr86_failure/check_builder_pipeline.py'
        old="(h/'tree_inverse.cuh').read_text()"
        new="'\\n'.join(function((h/'tree_inverse.cuh').read_text(),n) for n in ('__device__ __forceinline__ void qsb_block_inverse_tree_scratch(', '__device__ __forceinline__ void qsb_block_inverse_tree('))"
        root_wrapper='extern "C" int audit_roots(int count){\n int groups=(count+255)/256;std::vector<uint64_t> roots(count*4+1,0),super(groups*4+1,0),tree(groups*4*QSB_CHECKPOINT_STRIDE+1,0xcafe);\n roots.back()=0xbeef;super.back()=0xabcd;for(int i=0;i<count;i++)roots[4*i]=i+2;\n inverse_count=0;\n for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),count,super.data(),tree.data());});\n launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});\n for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),count,super.data(),tree.data());});\n require(inverse_count==1&&tree.back()==0xcafe&&roots.back()==0xbeef&&super.back()==0xabcd);\n BIGNUM*p=BN_new(),*v=BN_new(),*want=BN_new(),*got=BN_new();require(EC_GROUP_get_curve(group,p,nullptr,nullptr,ctx));\n for(int i=0;i<count;i++){\n  require(BN_set_word(v,i+2));require(BN_mod_inverse(want,v,p,ctx)!=nullptr);require(BN_lebin2bn((uint8_t*)&roots[4*i],32,got)!=nullptr);\n  if(BN_cmp(want,got)){fprintf(stderr,"ROOT_HIERARCHY_MISMATCH count=%d i=%d\\n",count,i);return 0;}\n }\n BN_free(p);BN_free(v);BN_free(want);BN_free(got);return 1;\n}\n'
        replace_wrapper=("code+='\\n'+wrapper", "code+='\\n'+wrapper+"+repr(root_wrapper))
        init_line="  lib.init((C.c_uint64*4)(*[coefficient>>(64*j)&((1<<64)-1) for j in range(4)]))"
        root_calls=init_line+"\n  if coefficient==official:\n   lib.audit_roots.argtypes=[C.c_int]\n   for root_count in [1,255,256,257,2047,2048]:assert lib.audit_roots(root_count),root_count"
        pipeline_projection=project_run(pipeline,[(old,new),replace_wrapper,(init_line,root_calls)],['--source',str(src),'--compact','--fused','--widths',widths,'--low-bits','8','--report',str(out/'checkpoint-pipeline.json')])
    end=source_identity(src)
    # All full-closure reports must bind one stable snapshot.
    assert end==start,'Candidate changed during builder checks; rerun after concurrent owners finish.'
    sizes={'table':(1<<19)*64,'host_L':16*256*64,'host_H':16*256*64,'device_L':16*256*64,'device_H':16*256*64,'roots':4096*32,'tree':4096*4*256*8,'super_roots':16*32,'root_tree':16*4*256*8}
    summary={'status':'PASS',**start,'checker_sha256':sha(Path(__file__)),'source_files_checked':{p.name:sha(p) for p in [h/'compact_geometry.cuh',h/'compact_table_device.cuh',h/'compact_table_kernels.cuh',h/'compact_table_host.cuh',h/'builder_checkpoint.cuh',h/'tree_inverse.cuh']},'adaptors':{'ladder_affine':{'path':str(wide.relative_to(ROOT)),'sha256':sha(wide),'projection_sha256':wide_projection,'change':'Execute decoder exhaustively for all524288entries; actual ladder/affine functions unchanged.'},'checkpoint_pipeline':{'path':'research/pr86_failure/check_builder_pipeline.py','projection_sha256':pipeline_projection,'change':'Include only original full-CTA inverse APIs used by builder; unused search192/224subgroup bodies omitted, actual startup checkpoint algorithms unchanged.'}},'descriptor_check':descriptor,'allocation_bytes':sizes,'peak_device_bytes':sum(v for k,v in sizes.items() if not k.startswith('host_')),'dispatch':{'chunk_capacity_entries':1<<20,'actual_table_entries':1<<19,'chunks':1,'root_hierarchy_counts_tested':[1,255,256,257,2047,2048],'prepare_finish_blocks':2048,'root_groups':8,'super_root_active_lanes':8,'root_blocks_capacity':4096,'group_capacity':16,'identity_participants':'Full256thread checkpoint blocks and root-inverse block; inactive super-root lanes useidentity.'},'bounds_proof':'For0<=t<2^19,ch=t>>15 is0..15,m=2*(t-(ch<<15))+1 isodd1..65535,hi=m>>8 is0..255,lo=m&255 isodd1..255. Lastentrywritesbytes[33554368,33554432), nooverflow. H[0]unused; otherH=hi*256base,L=lobase. Theirsum anddifference arenonzero modn becauseinteger0<hi*256±lo<65536<n andmultipliers2^(16ch),runtimecoef areinvertiblemodprimeorder.','host_review':'Highladder capacity checked. Every CUDA allocation/copy/launch/sync/free checked; device synchronization precedes scratchfree andspotcheck. Scratch remains1Mentry capacity intentionally; actual tablehalfchunk. Pointserialization andspot-check OpenSSL returns now checked. Spotcorners include127/128/129firsthigh-radixtransition.','results':{'ladder_affine':'builder-results/ladder-affine.json','checkpoint_pipeline':None if a.skip_pipeline else 'builder-results/checkpoint-pipeline.json'},'gpu_executed':False,'limits':'OpenSSL host field substitutions and CPU cooperative threads, not actual PTX arithmetic/CUDA execution. Full decoder domain andfullhostladders checked; selectedtableentries/hierarchycases, notall524288GPUtableoutputs. Source-lifetime review, notCUDAfaultinjection.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('BUILDER_SUMMARY',json.dumps({'status':'PASS','fingerprint':start['source_fingerprint'],'summary':str(out/'summary.json')}))
if __name__=='__main__':main()
