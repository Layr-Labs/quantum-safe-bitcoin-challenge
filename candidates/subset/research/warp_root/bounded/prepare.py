#!/usr/bin/env python3
"""Create isolated capped HM43 variant, preserving unbounded control and ready3ac."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'candidate'
SUBSET=HERE.parents[2]
sys.path.insert(0,str(SUBSET))
from preflight import source_identity
CONTROL_FP='6cb9b00606f541cd0894a48308fa9771dd68efe4bb269f49515e1f8e14d4435d'
READY=HERE.parents[1]/'weak_field/ordinary_region/candidate'
READY_FP='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
HEADER='tests/gpu_epochs/hm43_warp_inverse.cuh'
TREE='tests/gpu_epochs/tree_inverse.cuh'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(text,before,after):
 assert text.count(before)==1,repr(before)
 return text.replace(before,after,1)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--output',type=Path,default=HERE/'candidate',help='Fresh directory strictly below bounded/')
 args=ap.parse_args();out=args.output.resolve()
 assert out.is_relative_to(HERE.resolve()) and out!=HERE.resolve(),'Output must remain strictly below bounded/'
 assert not out.exists(),'Preserve existing candidate; output must not exist'
 old=source_identity(CONTROL);ready=source_identity(READY)
 assert old['source_fingerprint']==CONTROL_FP
 assert ready['source_fingerprint']==READY_FP
 transformations=[]
 text=(CONTROL/HEADER).read_text()
 edits=[('#endif\n#ifdef HM43_HOST_ORACLE', '''#endif
// A finite cooperative budget closes the fixed-width accumulator proof.
// Smaller limits are supported for forced-fallback tests; zero uses the baseline.
#ifndef HM43_WARP_MAX_BATCHES
#define HM43_WARP_MAX_BATCHES 16
#endif
#if HM43_WARP_MAX_BATCHES < 0 || HM43_WARP_MAX_BATCHES > 16
#error "HM43_WARP_MAX_BATCHES must be between 0 and 16"
#endif
#ifdef HM43_HOST_ORACLE'''),
 ('__device__ __forceinline__ void hm43_warp_inverse(uint64_t result[5],int lane){', '''// All32 lanes return the same completion flag. On false, result is untouched.
// True means completed arithmetic; noninvertible inputs retain the zero result.
__device__ __forceinline__ bool hm43_warp_inverse(uint64_t result[5],int lane){'''),
 ('    uint32_t nonzero=0;\n    while(true){\n        uint32_t nz=', '''    uint32_t nonzero=0;
    unsigned completed_batches=0;
    while(true){
        // Uniform return before any next-batch collective. No shared root write.
        if(completed_batches==HM43_WARP_MAX_BATCHES)return false;
        ++completed_batches;
        uint32_t nz='''),
 ('    for(int j=0;j<5;j++)result[j]=hm43_exchange(result[j],0);\n}', '    for(int j=0;j<5;j++)result[j]=hm43_exchange(result[j],0);\n    return true;\n}')]
 for before,after in edits:text=replace_once(text,before,after)
 reverse=text
 for before,after in reversed(edits):reverse=replace_once(reverse,after,before)
 assert reverse==(CONTROL/HEADER).read_text()
 tree=(CONTROL/TREE).read_text()
 before='''        hm43_warp_inverse(root,lane);
        if(ecid==0){
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][1]=root[k];
        }'''
 after='''        const bool root_complete=hm43_warp_inverse(root,lane);
        if(ecid==0){
            if(!root_complete){
                // Shared tree[1] is still the original canonical input. Never
                // invert a partial cooperative state; restore the full ABI.
                #pragma unroll
                for(int k=0;k<4;k++)root[k]=tree[k][1];
                root[4]=0;
                _ModInv(root);
            }
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][1]=root[k];
        }'''
 tree=replace_once(tree,before,after)
 assert replace_once(tree,after,before)==(CONTROL/TREE).read_text()
 for rel in list(old['source_sha256'])+['COPYING']:
  dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(CONTROL/rel,dst)
 (out/HEADER).write_text(text);(out/TREE).write_text(tree)
 current=source_identity(out);assert len(current['source_sha256'])==19
 changed=[p for p,h in old['source_sha256'].items() if current['source_sha256'][p]!=h]
 assert sorted(changed)==sorted([HEADER,TREE])
 assert source_identity(CONTROL)==old and source_identity(READY)==ready
 (out/'PROTOTYPE_NOT_QUALIFIED.md').write_text('Isolated bounded HM43 root experiment. Default16 complete matrix batches, then original scalar-root fallback. Source-only preparation: no CPU/native/GPU qualification yet. Preserve unbounded6cb and ready3ac. See bounded/prepared-source.json and proof-obligations.md.\n')
 d={'status':'ISOLATED_BOUNDED_HM43_PROTOTYPE_NOT_QUALIFIED','source_directory':str(out.relative_to(HERE)),**current,'control_source_fingerprint':CONTROL_FP,'ready_source_fingerprint':READY_FP,'default_complete_matrix_batch_cap':16,'test_cap_macro':'HM43_WARP_MAX_BATCHES','allowed_cap_values':[0,16],'helper_interface':'bool hm43_warp_inverse(uint64_t result[5],int lane)','return_contract':'Uniform true for completed arithmetic, false for cap. False leaves every caller result array untouched. True returns inverse/zero by original contract.','fallback':'Onlyecid0 reloadstree[0..3][1],clearsroot[4],runsunchanged_ModInv,thenpublishes4words. Sharedoriginalroot remainsuntouched untilpublication.','changed_files':changed,'header_reverse_patch_exact_control':True,'tree_reverse_patch_exact_control':True,'inherited_provenance':{'donor_head':'3c3d748e7e46bbf68fcd22d179c44b231eaa15de','donor_header_sha256':old['source_sha256'][HEADER],'coauthor':'AbdelStark','original_algorithm':'JeanLucPons/VanitySearch GPUMath.h, GPL-3.0-only notice retained'},'range_review_sha256':sha(HERE.parent/'range-review.json'),'generator_sha256':sha(Path(__file__)),'control_and_ready_hashes_preserved':True,'cpu_helper_or_tree_tested':False,'native_compiled':False,'gpu_executed':False,'submission_performed':False}
 (HERE/'prepared-source.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'status':d['status'],'source_fingerprint':current['source_fingerprint'],'closure_files':19,'header_sha256':current['source_sha256'][HEADER],'tree_sha256':current['source_sha256'][TREE]}))
if __name__=='__main__':main()
