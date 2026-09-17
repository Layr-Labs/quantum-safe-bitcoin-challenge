#!/usr/bin/env python3
"""Actual scoped-load packed EC192 chain/recovery CPU projection; no CUDA synchronization proof."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'research/wide_windows')]
from preflight import source_identity
from audit_support import function

def once(s, old, new):
    assert s.count(old) == 1, ('projection marker changed', old)
    return s.replace(old, new, 1)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source', type=Path, default=HERE/'candidate')
ap.add_argument('--output', type=Path, default=HERE/'field-check')
ap.add_argument('--mutation-packed-sign',action='store_true',help='Negative control: CPU projection only, erase packed consumer sign')
args = ap.parse_args()
source, output = args.source.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
identity = source_identity(source)
header = (source/'tests/gpu_epochs/compact_table_device.cuh').read_text()
tree = (source/'tests/gpu_epochs/tree.cu').read_text()
kernel = function(tree, '__global__ void __launch_bounds__(256,3) kernel_digest_specialist(')
call = 'compact_ec192_fixed_xyzz_packed_shared(qx,qy,qzz,qzzz,z,d_gt,nullptr,arena.ec.fields,ecid);'
assert kernel.count(call) == 1
for text in ['ecid=tid-64', 'uint64_t fields[4][768];',
             'qsb_xyzz_finish_prepare(qx,qzz,qzzz,u2rx,prod);',
             'qsb_xyzz_finish_precomputed(qzz,qy,Wsave,qzzz,prod,u2rx,u2ry,q1x,q2x);']:
    assert text in kernel, text

# The production path selects the packed helper, never the preserved raw helper.
assert 'compact_ec192_fixed_xyzz_shared(' not in kernel
assert 'qsb_pack_digest16(digest);' in kernel
base_header=(HERE.parents[1]/'small32/candidate/tests/gpu_epochs/compact_table_device.cuh').read_text()
names = {
    'compact_fixed_xyzz_shared': 'compact_ec192_fixed_xyzz_packed_shared',
    'qsb_shared_field_bits': 'qsb_ec192_shared_field_bits',
    'qsb_shared_direct_digit': 'qsb_ec192_shared_direct_digit',
    'qsb_PointAddXYZZ_shared_z_def': 'qsb_ec192_PointAddXYZZ_shared_z_def',
}
hashes={}
for spec in [
    '__device__ void qsb_ec192_PointAddXYZZ_shared_z_def(',
    '__device__ __forceinline__ void qsb_asym_last_add(',
    '__device__ __forceinline__ uint32_t gt_field_bits_v(',
    '__device__ __forceinline__ void gt_direct_digit(',
]:
    actual=function(header,spec)
    assert actual==function(base_header,spec),spec
    hashes[spec]=hashlib.sha256(actual.encode()).hexdigest()
packed_helpers='\n'.join(function(header,spec) for spec in [
    '__device__ __forceinline__ void qsb_pack_scalar16(',
    '__device__ __forceinline__ void qsb_ec192_packed_digit(',
])
if args.mutation_packed_sign:
    packed_helpers=once(packed_helpers,'*neg=cell>>15;','*neg=0; // deliberately invalid packed sign')
for spec in [
    '__device__ __forceinline__ void qsb_pack_scalar16(',
    '__device__ __forceinline__ void qsb_ec192_packed_digit(',
    '__device__ void compact_ec192_fixed_xyzz_packed_shared(',
]: hashes[spec]=hashlib.sha256(function(header,spec).encode()).hexdigest()

# Adapt independent geometry expectations and constructed exceptional scalars;
# no actual production chain statement is rewritten by this adapter.
def geometry_adapter(adapter):
    old_setup = """first=args.first_width;small_bits=first+11*17;last_shift=small_bits+26;last_width=256-last_shift;last_small_shift=small_bits-17
widths=[first]+[17]*11+[26,last_width]"""
    new_setup = """first=16;small_bits=224;last_shift=240;last_width=16;last_small_shift=208
widths=[16]*16"""
    transforms = [
        (old_setup,new_setup),
        ('for c in range(11):','for c in range(13):'),
        ('end=first+17*c;previous=0 if c==0 else first+17*(c-1)', 'previous=sum(widths[:c]);end=previous+widths[c]'),
        ('digit_max=((1<<(first if c==0 else 17))-1)<<previous', 'digit_max=((1<<widths[c])-1)<<previous'),
        ("'--prefetch-start','0','--prefetch-chunks','12'", "'--prefetch-start','0','--prefetch-chunks','14'"),
        ("'--chain-order','12,13,0,1,2,3,4,5,6,7,8,9,10,11'", "'--chain-order','14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13'"),
        ('cold_m=N-((1<<18)-2)*(1<<last_small_shift)', 'cold_m=N-((1<<17)-2)*(1<<last_small_shift)'),
        ('d11=(((cold_m>>last_small_shift)&((1<<18)-1))|1)-(1<<17)', 'd11=(((cold_m>>last_small_shift)&((1<<17)-1))|1)-(1<<16)'),
        ('assert d11==-(1<<17)+1', 'assert d11==-(1<<16)+1'),
        ('through window10. Final helper handles both remaining cases.', 'through window12. Final window13 is guarded; all-input proof is resident_tables/exception-design.md.'),
    ]
    code=''.join(f'    exceptional_original = replace_once(exceptional_original, {a!r}, {b!r})\n' for a,b in transforms)
    marker='    exceptional_original = exception_path.read_text()\n'
    adapter=once(adapter,marker,marker+code)
    adapter=once(adapter,"assert curve['geometry_bits'] == [17] * 12 + [26] * 2", "assert curve['geometry_bits'] == [16] * 16")
    adapter=once(adapter,"assert curve['checked_chain_order'] == [12, 13] + list(range(12))", "assert curve['checked_chain_order'] == [14, 15] + list(range(14))")
    adapter=once(adapter,"assert curve['table_bytes'] == (48 << 20) + (4 << 30)", "assert curve['table_bytes'] == (32 << 20)")
    adapter=once(adapter,'selected_bodies = [chain,', "selected_bodies = [chain, function(header, '__device__ void qsb_ec192_PointAddXYZZ_shared_z_def('),")
    adapter=once(adapter,"assert tree.count(call) == 1, ('Actual kernel/host handoff changed', call)", "assert call in tree, ('Actual handoff missing', call)")
    if args.mutation_packed_sign:
        adapter=adapter.replace('with contextlib.redirect_stdout(io.StringIO()):','with contextlib.nullcontext():')
    return adapter

legacy = ROOT/'research/shared_all_state/check.py'
outer = legacy.read_text()
start = outer.index('# Verify the inverse body itself')
end = outer.index('# Own-lane scratch ABI projection.')
# Kernel synchronization/inverse/SHA checks belong to the parent's specialist
# control checker, not this field oracle. Do not inherit generic-kernel claims.
outer = outer[:start] + outer[end:]
outer = once(outer, 'adapter=legacy.read_text()', 'adapter=geometry_adapter(legacy.read_text())')
for a,b in names.items():
    outer = outer.replace(a,b)
# Oracle starts from raw scalars. Execute the actual producer packer immediately
# before the exact consumer body in the CPU ABI wrapper, including in-place mode.
outer=once(outer,'# Own-lane scratch ABI projection.', 'helpers += '+repr('\n'+packed_helpers)+'\n# Own-lane scratch ABI projection.')
outer=once(outer,"projected=projected.replace(') {',", "projected=projected.replace('const uint64_t packed[4]','const uint64_t scalar[4]',1)\nprojected=projected.replace(') {',")
outer = outer.replace('arena[4][1024]', 'arena[4][768]')
outer = outer.replace('j<1024', 'j<768').replace('(j%256)', '(j%192)')
outer = outer.replace('(next_tid++*73u)&255u', '(next_tid++*73u)%192u')
outer = once(outer, '    unsigned tid=(next_tid++*73u)%192u;', '''    unsigned tid=(next_tid++*73u)%192u;
    uint64_t packed[4], packed_alias[4];
    qsb_pack_scalar16(scalar,packed);
    memcpy(packed_alias,scalar,32);qsb_pack_scalar16(packed_alias,packed_alias);
    require(memcmp(packed,packed_alias,32)==0);
    static unsigned owner_seen[192]={};owner_seen[tid]++;
    if(next_tid==192)for(unsigned owner=0;owner<192;owner++)require(owner_seen[owner]>0);''')
outer = once(outer, '    uint64_t arena[4][768];', '''    struct GuardedArena {uint64_t before[8], fields[4][768], after[8];} guard;
    uint64_t (&arena)[4][768]=guard.fields;
    for(int j=0;j<8;j++)guard.before[j]=guard.after[j]=0x9ad34e17b62105c8ULL;''')
outer = once(outer, '    for(int k=0;k<4;k++)for(unsigned j=0;j<768;j++)', '''    for(int j=0;j<8;j++)require(guard.before[j]==0x9ad34e17b62105c8ULL && guard.after[j]==0x9ad34e17b62105c8ULL);
    for(int k=0;k<4;k++)for(unsigned j=0;j<768;j++)''')
outer = outer.replace('shared_phase.fields,threadIdx.x', 'arena.ec.fields,ecid')
# Replace the legacy summary rather than claiming checks that were not run.
outer = outer[:outer.index("report={'status':'PASS'")]
sys.argv = [str(legacy),'--source',str(source),'--output',str(output)]
exec(compile(outer,str(legacy)+'[EC192-field-projection]','exec'),
     {'__name__':'__main__','__file__':str(legacy),'geometry_adapter':geometry_adapter})
assert source_identity(source) == identity, 'Production closure changed during run'
assert not args.mutation_packed_sign, 'Packed sign mutation unexpectedly passed actual chain oracle'
result = json.loads((output/'chain_recovery/chain-recovery-results.json').read_text())
assert result['curve_chains'] >= 512
assert result['recovered_keys_compared'] >= 1000
assert sum(result['actual_helper_cases'][k] for k in ['equal','opposite','ordinary']) == 477
assert result['actual_helper_cases']['unguarded_doubling_failures_detected'] == 159
assert result['prefetch_next_load_checks'] == result['curve_chains']*14
assert result['source_fingerprint'] == identity['source_fingerprint']
# Re-run the generalized integer-domain argument for this geometry, and bind
# every modeled primitive to the frozen source. Its inequalities cover all raw
# uint256 scalars; its samples only cross-check the integer/limb translations.
proof_path = HERE.parents[1]/'check_exception_design.py'
proof_spec = importlib.util.spec_from_file_location('resident_exception_design',proof_path)
proof_module = importlib.util.module_from_spec(proof_spec)
proof_spec.loader.exec_module(proof_module)
domain = proof_module.assess([16]*16)
base = ROOT/'research/specialist_warps/two_six_ring/candidate'
proof_helpers = {}
for relative,sig in [
    ('tests/gpu_epochs/tree.cu','__device__ __forceinline__ void gt_recode_setup('),
    ('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ uint32_t gt_field_bits_v('),
    ('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ void gt_direct_digit('),
    ('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ void qsb_asym_last_add('),
]:
    body=function((source/relative).read_text(),sig)
    assert body==function((base/relative).read_text(),sig),sig
    proof_helpers[sig]=hashlib.sha256(body.encode()).hexdigest()
# Explicit structural assumptions of the theorem; independent load-order and
# digit/address oracles above also execute these exact chain statements.
chain=function(header,'__device__ void compact_ec192_fixed_xyzz_packed_shared(')
for statement in [
    'qsb_ec192_packed_digit(arena,tid,14u,&idx,&neg);', 'qsb_ec192_packed_digit(arena,tid,15u,&idx,&neg);',
    'compact_load_signed(gTX,gTY,14,idx,neg,x0,y0);',
    'compact_load_signed(gTX,gTY,15,idx,neg,x1,y1);',
    'for(int i=0;i<13;++i)', 'int c=i;',
    'qsb_ec192_packed_digit(arena,tid,(unsigned)c,&idx,&neg);',
    'qsb_ec192_packed_digit(arena,tid,(unsigned)c+1u,&hint,&unused_sign);',
    'compact_prefetch_point(gTX,c+1,hint);',
    'qsb_ec192_packed_digit(arena,tid,13u,&idx,&neg);',
    'compact_load_signed(gTX,gTY,13,idx,neg,cx,cy);',
    'qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,anchor);',
]: assert statement in chain,statement
assert 'int next=c+1;' not in chain
assert chain.count('uint32_t hint; uint64_t unused_sign;')==2
assert 'compact_prefetch_point(gTX,0,hint);\n    }' in chain
assert 'compact_prefetch_point(gTX,c+1,hint);\n        }' in chain
N=proof_module.N
D=((1<<16)-1)<<208
extra=json.loads((output/'chain_recovery/exception-scalars.json').read_text())
extra={int(k,0) if isinstance(k,str) else k for k in extra}
assert {D,N-D} <= extra
wide_source=(ROOT/'research/wide_windows/check_wide.py').read_text()
assert 'chain_cases=sorted(edge)[::5]+[0,N,N-1,(1<<256)-1]' in wide_source
assert {int(k,0) for k in domain['complete_exception_raw_scalars']}=={0,N,D,N-D}
domain.update({'source_fingerprint':identity['source_fingerprint'],
    'proof_primitive_sha256':proof_helpers,
    'actual_chain_sha256':hashlib.sha256(chain.encode()).hexdigest(),
    'actual_pack_and_unpack_helpers_sha256':hashlib.sha256(packed_helpers.encode()).hexdigest(),
    'packing_equivalence_argument':'For window c=4k+j, f extracts bits16c+1..16c+16 of original M, with the higher-limb bit stitched at j=3. Let t=f>>15. Non-top idx=(f XOR(t-1))&32767 and neg=(t XOR1) XORglobal_sign reproduce the signed odd digit; top uses idx=f&32767 and global_sign. Cell=(idx|(neg<<15)) is recovered exactly by the consumer mask/shift. Rewriting M[k] is safe because later groups only read original M[k+1] and higher. This transfers the unchanged recoder domain theorem, supported by source-bound finite actual-chain execution; not exhaustive uint256 execution.',
    'all_four_final_exception_scalars_executed_by_actual_chain':True,
    'proof_script_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    'proof_scope':'Integer-domain inequalities and exhaustive final-digit equation, conditional on valid nonzero runtime base and correct field/table construction. Actual-source finite curve checks complement this proof; no GPU execution.'})
(output/'exception-domain-bound.json').write_text(json.dumps(domain,indent=2)+'\n')
assert source_identity(source)==identity, 'Production closure changed during proof binding'
report = {
    'status':'PASS', **identity, 'gpu_executed':False, 'production_modified':False,
    'scope':'Actual producer scalar packer and packed EC192 chain, packed-digit and shared-Z bodies executed with CPU-local guarded arena and OpenSSL field operations; independent curve/recovery/exception/address oracles retained. No native field arithmetic, CUDA scheduling, mailbox or full-kernel proof.',
    'recoding_cases':result['recoding_cases'], 'full_domain_bound_report':'exception-domain-bound.json',
    'all_four_final_exception_scalars_executed':True,
    'curve_chains':result['curve_chains'], 'recovered_keys_compared':result['recovered_keys_compared'],
    'actual_helper_cases':result['actual_helper_cases'], 'actual_loader_cases':result['actual_loader_cases'],
    'prefetch_next_load_checks':result['prefetch_next_load_checks'],
    'expected_prefetched_windows':list(range(14)), 'owner_count':192,
    'owner_sequence':'(call_index*73)%192; coprime step, all192 owners exercised and asserted within first192 chain calls',
    'canaries':'All non-owner arena cells and eight uint64 words before/after checked on every projected call',
    'arena_shape':[4,768], 'slot_offsets':[0,192,384,576],
    'selected_actual_packed_chain':True, 'fresh_current_digit_and_scoped_hint_structure':True, 'logical_packed_limb_reads_per_candidate':30, 'additional_logical_shared_read_bytes_vs_packed_baseline':112, 'actual_producer_scalar_packer_executed':True, 'raw_chain_used_as_production_test':False, 'packer_alias_comparisons':result['curve_chains'], 'geometry_bits':[16]*16, 'checked_chain_order':[14,15]+list(range(14)), 'table_bytes':32<<20, 'specialist_chain_and_recovery_handoffs_present':True,
    'generic_kernel_join_audit_used':False, 'actual_ec192_helpers_sha256':hashes,
    'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'adapted_outer_sha256':hashlib.sha256(outer.encode()).hexdigest(),
}
(output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','source_fingerprint','curve_chains','recovered_keys_compared','actual_helper_cases','owner_count']},indent=2))
