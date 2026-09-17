#!/usr/bin/env python3
"""Execute actual shared-state chain with existing independent curve oracles.
Only CPU projection adaptation occurs: one lane at a time, varied SoA owner,
canaries for other lanes, and an ABI-local arena. Not CUDA scheduling proof.
"""
import argparse,hashlib,json,pathlib,sys
sys.dont_write_bytecode=True
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research/wide_windows')]
from preflight import source_identity
from audit_support import function
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=pathlib.Path,default=HERE/'candidate');ap.add_argument('--output',type=pathlib.Path,default=HERE);args=ap.parse_args();BASE=args.source.resolve();OUTPUT=args.output.resolve();OUTPUT.mkdir(parents=True,exist_ok=True);DONOR=ROOT/'research/ranked_specialization/digits_candidate'
header=(BASE/'tests/gpu_epochs/compact_table_device.cuh').read_text(); tree=(BASE/'tests/gpu_epochs/tree.cu').read_text()
identity=source_identity(BASE);original=source_identity(DONOR)
old=function(header,'__device__ void compact_fixed_xyzz(')
actual=function(header,'__device__ void compact_fixed_xyzz_shared(')
helpers='\n'.join(function(header,s) for s in ['__device__ __forceinline__ uint32_t qsb_shared_field_bits(','__device__ __forceinline__ void qsb_shared_direct_digit(', '__device__ void qsb_PointAddXYZZ_shared_z_def('])
assert function(header,'__device__ __forceinline__ void qsb_asym_last_add(')==function((DONOR/'tests/gpu_epochs/compact_table_device.cuh').read_text(),'__device__ __forceinline__ void qsb_asym_last_add(')
# Verify the inverse body itself is byte-identical after only ABI/storage move.
inv=(BASE/'tests/gpu_epochs/tree_inverse.cuh').read_text();donor_inv=(DONOR/'tests/gpu_epochs/tree_inverse.cuh').read_text()
scratch=function(inv,'__device__ __forceinline__ void qsb_block_inverse_tree_scratch(')
expected=function(donor_inv,'__device__ __forceinline__ void qsb_block_inverse_tree(').replace('qsb_block_inverse_tree(uint64_t *value)','qsb_block_inverse_tree_scratch(uint64_t *value, uint64_t tree[4][512])',1).replace('    __shared__ uint64_t tree[4][512];\n','',1)
assert scratch==expected
kernel=function(tree,'__global__ void __launch_bounds__(256, 3) kernel_digest(')
def join_audit(k):
    before,after=k.split('qsb_block_inverse_tree_scratch(prod,shared_phase.inverse);')
    assert before.rstrip().endswith('__syncthreads();')
    assert 'if(!usable)return;' not in before and 'if(!usable)return;' in after
    assert 'compact_fixed_xyzz_shared(qx,qy,qzz,qzzz,z,d_gt,nullptr,shared_phase.fields,threadIdx.x);' in before
    assert before.count('__shared__ QsbSharedPhaseArena shared_phase;')==1
join_audit(kernel)
try:join_audit(kernel.replace('    __syncthreads();\n    qsb_block_inverse_tree_scratch','    qsb_block_inverse_tree_scratch'));raise RuntimeError('missing join accepted')
except AssertionError:pass
# Source-equivalent scheduled SHA scratch ABI, with no additional declaration.
schedule=(BASE/'tests/gpu_epochs/window_schedule_shared.cuh').read_text()
donor_schedule=(DONOR/'tests/gpu_epochs/window_schedule_shared.cuh').read_text()
sha_orig=function(donor_schedule,'__device__ __forceinline__ void qsb_scheduled_window_hash(')
sha_actual=function(schedule,'__device__ __forceinline__ void qsb_scheduled_window_hash_scratch(')
sha_expected=sha_orig.replace('qsb_scheduled_window_hash(','qsb_scheduled_window_hash_scratch(',1).replace('const epoch_desc_t *epoch, int lane) {','const epoch_desc_t *epoch, int lane, uint32_t first_states[8][256]) {',1).replace('    __shared__ uint32_t first_states[8][256];\n','',1)
assert sha_actual==sha_expected
chain_call='compact_fixed_xyzz_shared(qx,qy,qzz,qzzz,z,d_gt,nullptr,shared_phase.fields,threadIdx.x);'
def sha_join_audit(k):
    prefix=k.split(chain_call)[0]
    assert prefix.rstrip().endswith('__syncthreads();')
    assert prefix.count('return;')==1
    assert 'if(blockIdx.x*blockDim.x>=batch_size)return;' in prefix
sha_join_audit(kernel)
try:
    sha_join_audit(kernel.replace('__syncthreads();\n    '+chain_call,chain_call))
    raise RuntimeError('missing SHA reuse join accepted')
except AssertionError:pass
assert 'uint32_t sha[8][256];' in kernel and 'uint64_t fields[4][1024];' in kernel and 'uint64_t inverse[4][512];' in kernel
zbase=ROOT/'research/shared_z_state/candidate'
zsource=(zbase/'tests/gpu_epochs/compact_table_device.cuh').read_text()
zexpected=function(zsource,'__device__ void qsb_PointAddXYZZ_shared_z_def(').replace('arena[4][512]','arena[4][1024]').replace('arena[k][256+tid]','arena[k][768+tid]').replace('arena[k][tid]','arena[k][512+tid]')
assert function(header,'__device__ void qsb_PointAddXYZZ_shared_z_def(')==zexpected
# Own-lane scratch ABI projection. No chain statements are removed or replaced.
projected=actual.replace('compact_fixed_xyzz_shared(','compact_fixed_xyzz(',1).replace(', volatile uint64_t arena[4][1024], unsigned tid) {',') {',1)
projected=projected.replace(') {', ''') {
    static unsigned next_tid=0;
    unsigned tid=(next_tid++*73u)&255u;
    uint64_t arena[4][1024];
    for(int k=0;k<4;k++)for(int j=0;j<1024;j++)arena[k][j]=0x9ad34e17b62105c8ULL;
''',1)
projected=projected[:-1]+'''
    for(int k=0;k<4;k++)for(unsigned j=0;j<1024;j++)
        if((j%256)!=tid)require(arena[k][j]==0x9ad34e17b62105c8ULL);
}
'''
legacy=ROOT/'research/pr120_digits/check_chain_recovery.py';adapter=legacy.read_text()
def once(s,a,b):
 assert s.count(a)==1,('projection marker changed',a)
 return s.replace(a,b,1)
adapter=once(adapter,"chain = function(header, '__device__ void compact_fixed_xyzz(')","chain = function(header, '__device__ void compact_fixed_xyzz_shared(')")
adapter=once(adapter,"'_FixedBaseSignedXYZZStream(qx,qy,qzz,qzzz,z,d_gt);'","'compact_fixed_xyzz_shared(qx,qy,qzz,qzzz,z,d_gt,nullptr,shared_phase.fields,threadIdx.x);'")
adapter=once(adapter, "assert re.search(r'_PointAddXYZZ_def\\([^;]*,\\s*true\\s*\\)', chain)",
             "assert 'qsb_PointAddXYZZ_shared_z_def(X,Y,arena,tid,cx,cy,anchor,true);' in chain")
# The wide oracle reads actual source itself; replace ONLY which body its existing
# seven-argument call invokes, retaining every statement of the new shared body.
marker='    # Bind helper call arities to the selected source, so projections cannot'
extra="    wide = replace_once(wide, \"        compact=compact.replace('#include \\\"compact_geometry.cuh\\\"','')\", \"        compact=compact.replace(\" + repr("+repr(old)+") + \",\" + repr("+repr(projected)+") + \")\\n\" + \"        compact=compact.replace('#include \\\"compact_geometry.cuh\\\"','')\")\n"
adapter=once(adapter,marker,extra+marker)
# Inject actual shared helpers alongside existing actual direct-digit helpers.
needle="constant_mock + host_constant + function(header, '__device__ __forceinline__ uint32_t gt_field_bits_v(')"
adapter=once(adapter,needle,"constant_mock + host_constant + "+repr(helpers)+" + function(header, '__device__ __forceinline__ uint32_t gt_field_bits_v(')")
adapter=adapter.replace("'compact_fixed_xyzz': chain,","'compact_fixed_xyzz_shared': chain,")
sys.argv=[str(legacy),'--source',str(BASE),'--output',str(OUTPUT/'chain_recovery')]
exec(compile(adapter,str(legacy)+'[shared-state-projection]','exec'),{'__name__':'__main__','__file__':str(legacy)})
assert source_identity(BASE)==identity
report={'status':'PASS',**identity,'projection':'Actual shared helper bodies and every shared chain statement execute; only ABI gains CPU-local32KiB arena and varied owner tid. Independent OpenSSL oracle and sparse vector-loader checks unchanged. No GPU execution.','inverse_scratch_body_equal_base':True,'final_guard_equal_base':True,'uniform_join_audit':True,'sha_scratch_body_equal_base':True,'z_helper_equal_checked_z_variant_with_slot_remap':True,'explicit_view_strides':[256,1024,512],'missing_join_mutation_rejected':True,'missing_sha_join_mutation_rejected':True,'other_lane_arena_canaries':True,'checked_owner_sequence':'tid=(call_index*73)&255 covers all256lanes','actual_shared_chain_sha256':hashlib.sha256(actual.encode()).hexdigest(),'actual_shared_digit_helpers_sha256':hashlib.sha256(helpers.encode()).hexdigest(),'adapter_sha256':hashlib.sha256(adapter.encode()).hexdigest(),'gpu_executed':False}
(OUTPUT/'check-results.json').write_text(json.dumps(report,indent=2)+'\n')
