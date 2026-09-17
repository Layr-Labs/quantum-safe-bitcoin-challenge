#!/usr/bin/env python3
"""Actual weak ordinary region in the complete packed chain; independent EC oracle.
The inherited OpenSSL backend remains only for canonical seed/final/recovery and
reference curve operations. Weak add/sub/mul/square/normalize use actual host code.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/wide_windows')]
from preflight import source_identity
from audit_support import function

def once(s,a,b):
    assert s.count(a)==1,('adapter marker changed',a)
    return s.replace(a,b,1)

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,default=HERE/'candidate')
ap.add_argument('--output',type=Path,default=HERE/'chain-check')
ap.add_argument('--mutation-weak-zz',action='store_true',help='CPU projection only: erase actual weak helper shared ZZ update')
a=ap.parse_args();source=a.source.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
identity=source_identity(source)
legacy=ROOT/'research/specialist_warps/resident_tables/packed_digits/scoped_load/check_field.py'
base=legacy.parent/'candidate'
header=(source/'tests/gpu_epochs/compact_table_device.cuh').read_text();oldheader=(base/'tests/gpu_epochs/compact_table_device.cuh').read_text()
weak_sig='__device__ void qsb_ec192_PointAddXYZZ_shared_z_weak('
canonical_sig='__device__ void qsb_ec192_PointAddXYZZ_shared_z_def('
weak=function(header,weak_sig)
expected=function(oldheader,canonical_sig).replace('qsb_ec192_PointAddXYZZ_shared_z_def(','qsb_ec192_PointAddXYZZ_shared_z_weak(')
expected=expected.replace('_ModMult(','qsb_weak_mul(').replace('_ModSqr(','qsb_weak_square(').replace('_ModSub256(','qsb_weak_sub(')
expected=once(expected,'_ModAdd256(T, T, PPP);','qsb_weak_add(T, T, PPP);')
assert weak==expected,'Weak helper includes changes beyond the qualified substitutions'
chain_sig='__device__ void compact_ec192_fixed_xyzz_packed_shared('
chain=function(header,chain_sig);oldchain=function(oldheader,chain_sig)
expected_chain=once(oldchain,'qsb_ec192_PointAddXYZZ_shared_z_def(','qsb_ec192_PointAddXYZZ_shared_z_weak(')
expected_chain=once(expected_chain,'    qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,anchor);','''    // Restore the original canonical contract at the region boundary.
    qsb_weak_normalize(X); qsb_weak_normalize(Y);
    qsb_weak_normalize(ZZ); qsb_weak_normalize(ZZZ);
    qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,anchor);''')
assert chain==expected_chain
assert chain.count('qsb_weak_normalize(')==4 and 'qsb_ec192_PointAddXYZZ_shared_z_def(' not in chain
preserved={}
for path,sig in [('GPUMath.h','__device__ void _PointAddXYZZ_mm_def('),('tests/gpu_epochs/compact_table_device.cuh','__device__ __forceinline__ void qsb_asym_last_add('),('tests/gpu_epochs/tree.cu','__device__ __forceinline__ void qsb_xyzz_finish_prepare('),('tests/gpu_epochs/tree.cu','__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')]:
    body=function((source/path).read_text(),sig);assert body==function((base/path).read_text(),sig);preserved[sig]=hashlib.sha256(body.encode()).hexdigest()
addsub=(source/'qsb_weak_addsub.cuh').read_text();product=(source/'qsb_weak_product.cuh').read_text()
assert hashlib.sha256(addsub.encode()).hexdigest()=='31f72e474a8e42ae291b71a4f7250a3bc43c489c367a9787797f74169c52c88e'
# Actual complete headers select their ordinary C++ host branches because the
# CPU projection does not define __CUDA_ARCH__. No weak arithmetic is mocked.
weak_headers=addsub+'\n'+product+'\n'
for name in ['qsb_weak_add','qsb_weak_sub','qsb_weak_normalize','qsb_weak_mul','qsb_weak_square']:
    assert name+'(' in weak_headers
projected_weak=weak
if a.mutation_weak_zz:
    projected_weak=once(weak,'arena[k][384+tid]=ZZ1[k];','arena[k][384+tid]=arena[k][384+tid]; // invalid omission')
text=legacy.read_text()
text=text.replace('qsb_ec192_PointAddXYZZ_shared_z_def','qsb_ec192_PointAddXYZZ_shared_z_weak')
text=once(text,'assert actual==function(base_header,spec),spec',"assert spec == weak_sig or actual==function(base_header,spec),spec")
# Keep the old arithmetic-source equality check meaningful via expected above.
# The shared-all-state adaptor assembles helper text before its ABI projection.
marker="outer = once(outer, 'adapter=legacy.read_text()', 'adapter=geometry_adapter(legacy.read_text())')"
extra="\nouter = once(outer, '# Own-lane scratch ABI projection.', 'helpers = '+repr(weak_headers)+'+helpers\\n# Own-lane scratch ABI projection.')\n"
# Injection happens before names remapping, selecting the weak helper afterwards.
text=once(text,marker,marker+extra)
# In-memory helper mutation only. Other retained raw/canonical functions and
# independent OpenSSL expectations stay unchanged.
marker2="outer = outer.replace('shared_phase.fields,threadIdx.x', 'arena.ec.fields,ecid')"
if a.mutation_weak_zz:
    text=once(text,marker2,marker2+"\nouter=once(outer, '# Own-lane scratch ABI projection.', 'helpers=helpers.replace('+repr(weak)+','+repr(projected_weak)+')\\n# Own-lane scratch ABI projection.')")
    text=once(text,'    return adapter\n',"    adapter=adapter.replace('with contextlib.redirect_stdout(io.StringIO()):','with contextlib.nullcontext():')\n    return adapter\n")
sys.argv=[str(legacy),'--source',str(source),'--output',str(output)]
ns={'__name__':'__main__','__file__':str(legacy),'weak_sig':weak_sig,'weak_headers':weak_headers,'weak':weak,'projected_weak':projected_weak}
exec(compile(text,str(legacy)+'[actual-weak-region-adapter]','exec'),ns)
assert not a.mutation_weak_zz,'Weak ZZ mutation unexpectedly passed independent chain'
assert source_identity(source)==identity
report=json.loads((output/'results.json').read_text())
report.update({'status':'PASS_ACTUAL_WEAK_HOST_CHAIN_AND_CANONICAL_BOUNDARIES','scope':__doc__,
 'new_weak_helper_executed':True,'canonical_reference_helper_not_selected':True,
 'actual_weak_primitive_host_bodies_executed':['qsb_weak_add','qsb_weak_sub','qsb_weak_mul','qsb_weak_square','qsb_weak_normalize'],
 'host_square_uses_actual_weak_mul_fallback':True,'weak_primitives_OpenSSL_mocked':False,
 'normalizations_per_chain':4,'total_executed_boundary_normalizations':4*report['curve_chains'],
 'ordinary_weak_additions_per_chain':13,'actual_weak_helper_sha256':hashlib.sha256(weak.encode()).hexdigest(),
 'actual_weak_addsub_sha256':hashlib.sha256(addsub.encode()).hexdigest(),
 'actual_weak_product_sha256':hashlib.sha256(product.encode()).hexdigest(),
 'preserved_canonical_function_sha256':preserved,
 'legacy_scoped_checker_sha256':hashlib.sha256(legacy.read_bytes()).hexdigest(),
 'weak_adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'adapted_checker_sha256':hashlib.sha256(text.encode()).hexdigest(),
 'limits':'Actual CPU weak arithmetic in all13 ordinary additions and all4 boundary normalization calls; OpenSSL canonical seed/final/recovery and independent curve oracle. Does not execute product PTX, CUDA memory/scheduling or GPU kernels. Full-width primitive/PTX and helper-domain proofs are separate source-bound evidence.'})
(output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','source_fingerprint','curve_chains','recovered_keys_compared','owner_count','total_executed_boundary_normalizations']},indent=2))
