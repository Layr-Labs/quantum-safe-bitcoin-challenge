#!/usr/bin/env python3
"""Apply actual-chain/cubic oracles to13-window geometry; preserve oldcheckers."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

BASE = HERE/'thirteen_candidate'
OUTPUT = HERE/'thirteen_checks'
OUTPUT.mkdir(exist_ok=True)
ident = source_identity(BASE)
assert ident['source_fingerprint'] == '5d17e5fad6cb9d632d7b8b4977751da574242488d428cfedfb455accaa21ca5b'
legacy = ROOT/'research/pr120_digits/check_chain_recovery.py'
adapter = legacy.read_text()


def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new, 1)


# The legacy adapter already changes only extraction/ABI plumbing, retaining
# actual source bodies and independent OpenSSL arithmetic and curve oracles.
# This additional adapter changes only its independently expected geometry and
# the mathematical exception-bound setup, never the candidate source.
old_setup = '''first=args.first_width;small_bits=first+11*17;last_shift=small_bits+26;last_width=256-last_shift;last_small_shift=small_bits-17
widths=[first]+[17]*11+[26,last_width]'''
new_setup = '''first=18;small_bits=156;last_shift=231;last_width=25;last_small_shift=139
widths=[18]*3+[17]*6+[25]*4'''
transforms = [
    (old_setup, new_setup),
    ('for c in range(11):', 'for c in range(8):'),
    ('end=first+17*c;previous=0 if c==0 else first+17*(c-1)',
     'previous=sum(widths[:c]);end=previous+widths[c]'),
    ('digit_max=((1<<(first if c==0 else 17))-1)<<previous',
     'digit_max=((1<<widths[c])-1)<<previous'),
    ("'--prefetch-start','0','--prefetch-chunks','12'", "'--prefetch-start','0','--prefetch-chunks','9'"),
    ("'--chain-order','12,13,0,1,2,3,4,5,6,7,8,9,10,11'", "'--chain-order','9,10,11,12,0,1,2,3,4,5,6,7,8'"),
    ('through window10. Final helper handles both remaining cases.',
     'through small window7. Final window8 helper handles both remaining cases; four cold terms are separately bounded in exception_domain.py.'),
]
code = '\n'
for old, new in transforms:
    code += f'    exceptional_original = replace_once(exceptional_original, {old!r}, {new!r})\n'
adapter = once(adapter, '    exceptional_original = exception_path.read_text()\n',
               '    exceptional_original = exception_path.read_text()\n'+code)
adapter = once(adapter, "assert curve['geometry_bits'] == [17] * 12 + [26] * 2",
               "assert curve['geometry_bits'] == [18] * 3 + [17] * 6 + [25] * 4")
adapter = once(adapter, "assert curve['checked_chain_order'] == [12, 13] + list(range(12))",
               "assert curve['checked_chain_order'] == [9, 10, 11, 12] + list(range(9))")
sys.argv = [str(legacy), '--source', str(BASE), '--output', str(OUTPUT)]
exec(compile(adapter, str(legacy)+'[thirteen-geometry]', 'exec'),
     {'__name__':'__main__','__file__':str(legacy)})
assert source_identity(BASE) == ident
p = OUTPUT/'chain-recovery-results.json'
result = json.loads(p.read_text())
result['geometry_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['geometry_adapter_in_memory_sha256'] = hashlib.sha256(adapter.encode()).hexdigest()
result['geometry_adaptation'] = 'Only independent width/order/prefetch expectations and hot-prefix integer-bound setup change; all selected candidate chain/field/recovery bodies remain actual extracted source.'
p.write_text(json.dumps(result,indent=2)+'\n')
