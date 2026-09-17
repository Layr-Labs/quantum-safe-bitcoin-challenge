#!/usr/bin/env python3
"""Create immutable isolated HM43/EC192 candidate; no qualification or upload."""
from pathlib import Path
import hashlib,json,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
BASE=HERE.parent/'weak_field/ordinary_region/candidate'
DONOR=HERE.parent/'pending_sep17/pr189/head/tests/gpu_epochs/hm43_warp_inverse.cuh'
BASE_FP='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
DONOR_SHA='940f4c6ad4272a71d38533123a809c8f1affc7a9e0beca959db48380f24cd1fc'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 identity=source_identity(BASE);assert identity['source_fingerprint']==BASE_FP
 assert sha(DONOR)==DONOR_SHA
 out=HERE/'candidate';assert not out.exists(),'Preserve existing experiment; do not overwrite'
 for rel in list(identity['source_sha256'])+['COPYING']:
  dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/rel,dest)
 tree=out/'tests/gpu_epochs/tree_inverse.cuh';text=tree.read_text()
 sig='__device__ __forceinline__ void qsb_ec192_inverse_tree_scratch('
 start=text.index(sig);opening=text.index('{',start);depth=0;end=None
 for i in range(opening,len(text)):
  depth+=(text[i]=='{')-(text[i]=='}')
  if depth==0:end=i+1;break
 body=text[start:end]
 before='''    if(tid==0){
        uint64_t root[5]={0,0,0,0,0};
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=tree[k][1];
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][1]=root[k];
    }'''
 after='''    // Only the first EC warp (physical threads64..95) cooperates at the root.
    // All32 lanes, including HM43 guard lanes, execute every full-mask exchange.
    if(ecid<32){
        uint64_t root[5]={0,0,0,0,0};
        if(ecid==0){
            #pragma unroll
            for(int k=0;k<4;k++)root[k]=tree[k][1];
        }
        hm43_warp_inverse(root,lane);
        if(ecid==0){
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][1]=root[k];
        }
    }'''
 assert body.count(before)==1
 changed=body.replace(before,after);text=text[:start]+changed+text[end:]
 assert text.count('#pragma once')==1;text=text.replace('#pragma once','#pragma once\n#include "hm43_warp_inverse.cuh"',1);tree.write_text(text)
 shutil.copy2(DONOR,out/'tests/gpu_epochs/hm43_warp_inverse.cuh')
 new=source_identity(out);assert len(new['source_sha256'])==19
 for rel,h in identity['source_sha256'].items():
  if rel!='tests/gpu_epochs/tree_inverse.cuh':assert new['source_sha256'][rel]==h
 restored=tree.read_text().replace('#include "hm43_warp_inverse.cuh"\n','',1).replace(after,before,1)
 assert restored==(BASE/'tests/gpu_epochs/tree_inverse.cuh').read_text()
 report={'status':'ISOLATED_PROTOTYPE_NOT_QUALIFIED','base_source_fingerprint':BASE_FP,**new,'donor_head':'3c3d748e7e46bbf68fcd22d179c44b231eaa15de','donor_header_sha256':DONOR_SHA,'coauthor':'AbdelStark','generator_sha256':sha(Path(__file__)),'changes':['OnlyEC192rootbranch becomes firstfullECwarp cooperation','IncludeexactHM43header; correctedhotmath andweakordinaryregion unchanged'],'protected_ready_stage_changed':False,'gpu_executed':False,'cpu_helper_or_tree_tested':False,'native_compiled':False}
 (HERE/'prepared-source.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'PROTOTYPE_NOT_QUALIFIED.md').write_text('Isolated HM43 root-only experiment. No CPU/native/GPU qualification yet. Preserve ready3ac and PR179. See ../prepared-source.json.\n')
 print(json.dumps({'status':report['status'],'source_fingerprint':new['source_fingerprint'],'closure_files':len(new['source_sha256'])}))
if __name__=='__main__':main()
