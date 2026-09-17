#!/usr/bin/env python3
"""Compose existing actual shared-chain projection with checked13 expectations."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from preflight import source_identity


def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new, 1)


def geometry_adapter(adapter):
    old_setup = '''first=args.first_width;small_bits=first+11*17;last_shift=small_bits+26;last_width=256-last_shift;last_small_shift=small_bits-17
widths=[first]+[17]*11+[26,last_width]'''
    new_setup = '''first=18;small_bits=156;last_shift=231;last_width=25;last_small_shift=139
widths=[18]*3+[17]*6+[25]*4'''
    transforms = [
        (old_setup, new_setup),
        ('for c in range(11):', 'for c in range(8):'),
        ('end=first+17*c;previous=0 if c==0 else first+17*(c-1)', 'previous=sum(widths[:c]);end=previous+widths[c]'),
        ('digit_max=((1<<(first if c==0 else 17))-1)<<previous', 'digit_max=((1<<widths[c])-1)<<previous'),
        ("'--prefetch-start','0','--prefetch-chunks','12'", "'--prefetch-start','0','--prefetch-chunks','9'"),
        ("'--chain-order','12,13,0,1,2,3,4,5,6,7,8,9,10,11'", "'--chain-order','9,10,11,12,0,1,2,3,4,5,6,7,8'"),
        ('through window10. Final helper handles both remaining cases.', 'through window7. Final window8 is guarded; four cold terms are separately proved in geometry_frontier/exception_domain.md.'),
    ]
    code = ''.join(f'    exceptional_original = replace_once(exceptional_original, {a!r}, {b!r})\n' for a,b in transforms)
    adapter = once(adapter, '    exceptional_original = exception_path.read_text()\n', '    exceptional_original = exception_path.read_text()\n'+code)
    adapter = once(adapter, "assert curve['geometry_bits'] == [17] * 12 + [26] * 2", "assert curve['geometry_bits'] == [18] * 3 + [17] * 6 + [25] * 4")
    adapter = once(adapter, "assert curve['checked_chain_order'] == [12, 13] + list(range(12))", "assert curve['checked_chain_order'] == [9, 10, 11, 12] + list(range(9))")
    # Also bind the actual shared-Z body to actual GPUMath call arities.
    adapter = once(adapter, 'selected_bodies = [chain,', "selected_bodies = [chain, function(header, '__device__ void qsb_PointAddXYZZ_shared_z_def('),")
    return adapter


source = HERE/'candidate'
identity = source_identity(source)
assert identity['source_fingerprint'] == 'e80c8d9124fdd85ab38a6079ead00969d1e60d612a5f28c72f28523d28177908'
legacy = ROOT/'research/shared_all_state/check.py'
outer = legacy.read_text()
outer = once(outer, 'adapter=legacy.read_text()', 'adapter=geometry_adapter(legacy.read_text())')
sys.argv = [str(legacy), '--source', str(source), '--output', str(HERE)]
exec(compile(outer, str(legacy)+'[thirteen-shared-projection]', 'exec'),
     {'__name__':'__main__', '__file__':str(legacy), 'geometry_adapter':geometry_adapter})
assert source_identity(source) == identity
p = HERE/'check-results.json'
r = json.loads(p.read_text())
r['geometry_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
r['outer_adapter_sha256'] = hashlib.sha256(outer.encode()).hexdigest()
r['geometry_bits'] = [18]*3+[17]*6+[25]*4
r['checked_chain_order'] = [9,10,11,12]+list(range(9))
r['geometry_adaptation'] = 'Only independent geometry/order/prefetch expectations and prefix-bound setup changed. All actual shared chain/helper statements execute, with CPU-local arena and varied owner tid; no source mutation.'
p.write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps({'status':r['status'], 'source_fingerprint':r['source_fingerprint']}, indent=2))
