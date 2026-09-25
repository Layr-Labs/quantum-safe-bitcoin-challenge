#!/usr/bin/env python3
"""Narrow, fail-closed sm_89 load-order experiment inspired by CuAsmRL.

No assembler or RL runtime is required. Only a four-instruction, straight-line
bundle is permuted; every other byte of the ELF is retained. The CUDA compiler
still creates all instructions, relocations and resource metadata.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

CONTROL = 0x1FFFF << 41


def sections(blob):
    if blob[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected ELF64 little endian')
    if struct.unpack_from('<H', blob, 18)[0] != 190:
        raise ValueError('expected CUDA ELF')
    if struct.unpack_from('<I', blob, 48)[0] & 255 != 89:
        raise ValueError('expected sm_89')
    offset = struct.unpack_from('<Q', blob, 40)[0]
    size, count, string_index = struct.unpack_from('<HHH', blob, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', blob, offset+i*size)
               for i in range(count)]
    strings = headers[string_index]
    names = blob[strings[4]:strings[4]+strings[5]]
    return {names[h[0]:].split(b'\0')[0].decode(): (h[4], h[5]) for h in headers}


def controls(word):
    c = (word >> 41) & 0x1FFFF
    return dict(stall=c & 15, yield_bit=(c >> 4) & 1,
                write=(c >> 5) & 7, read=(c >> 8) & 7, wait=(c >> 11) & 63)


def validate_bundle(bundle):
    """Require exactly one immutable 64-byte table record and four unique tags."""
    if len(bundle) != 4:
        raise ValueError('four loads required')
    if [x['offset'] for x in bundle] != [0, 16, 32, 48]:
        raise ValueError('unexpected record offsets')
    if len({x['base'] for x in bundle}) != 1:
        raise ValueError('different address registers')
    base = bundle[0]['base']
    destinations = [{*range(x['dest'], x['dest']+4)} for x in bundle]
    if any(x['dest'] % 4 for x in bundle):
        raise ValueError('unaligned destination tuple')
    if len(set.union(*destinations)) != 16:
        raise ValueError('overlapping destination tuples')
    if set.union(*destinations) & {base, base+1}:
        raise ValueError('address overwritten by a load')
    cs = [controls(x['hi']) for x in bundle]
    if any(c['wait'] for c in cs):
        raise ValueError('incoming scoreboard wait')
    if len({c['write'] for c in cs}) != 4 or any(c['write'] == 7 for c in cs):
        raise ValueError('non-unique completion barriers')
    if cs[0]['read'] != 7 or len({c['read'] for c in cs[1:]}) != 1 or cs[1]['read'] == 7:
        raise ValueError('unexpected address read-barrier pattern')
    if any(x['hi'] >> 58 for x in bundle):
        raise ValueError('reuse flags or unknown high bits')
    if any(c['stall'] < 4 for c in cs[:3]):
        raise ValueError('unexpected issue spacing')


def patch(blob, sass):
    table = sections(blob)
    fn = None
    candidates = []
    run = []
    # cuobjdump, unlike nvdisasm, prints the ordinary Rbase.64 spelling.
    instruction = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/$')
    load = re.compile(r'LDG\.E\.128\.CONSTANT R(\d+), \[R(\d+)\.64(?:\+0x([0-9a-f]+))?\]')
    lines = sass.splitlines()
    for i, line in enumerate(lines):
        f = re.search(r'Function : (\S+)', line)
        if f:
            fn = f[1]
            run = []
        m = instruction.match(line)
        if not m or not fn or not fn.startswith('_Z23kernel_pinning_pipelineILb1ELi0EE'):
            continue
        lm = load.fullmatch(m[2].strip())
        if not lm:
            run = []
            continue
        addr = int(m[1], 16)
        if run and addr != run[-1]['address']+16:
            run = []
        section_offset, section_size = table['.text.'+fn]
        if addr+16 > section_size:
            raise ValueError('instruction outside section')
        lo, hi = struct.unpack_from('<QQ', blob, section_offset+addr)
        if lo != int(m[3], 16):
            raise ValueError('disassembly does not match binary')
        hi_match = re.fullmatch(r'\s*/\* 0x([0-9a-f]{16}) \*/', lines[i+1])
        if not hi_match or hi != int(hi_match[1], 16):
            raise ValueError('high instruction word mismatch')
        run.append(dict(address=addr, file_offset=section_offset+addr, lo=lo, hi=hi,
                        dest=int(lm[1]), base=int(lm[2]), offset=int(lm[3] or '0', 16),
                        text=m[2].strip(), function=fn))
        if len(run) >= 4:
            group = run[-4:]
            try:
                validate_bundle(group)
            except ValueError:
                continue
            candidates.append(group)
    if len(candidates) != 1:
        raise ValueError(f'expected one hot-loop bundle, found {len(candidates)}')
    group = candidates[0]
    result = bytearray(blob)
    permutation = [2, 3, 0, 1]  # Y's two chunks before X's two chunks.
    for position, origin in enumerate(permutation):
        old = group[origin]
        # Destination completion tags travel with their loads. Issue spacing,
        # yield, and address read-barrier positions remain the compiler's.
        control = (group[position]['hi'] >> 41) & 0x1FFFF
        control = (control & ~(7 << 5)) | (controls(old['hi'])['write'] << 5)
        hi = (old['hi'] & ~CONTROL) | (control << 41)
        struct.pack_into('<QQ', result, group[position]['file_offset'], old['lo'], hi)
    allowed = set(range(group[0]['file_offset'], group[-1]['file_offset']+16))
    changed = [i for i, (a, b) in enumerate(zip(blob, result)) if a != b]
    if not changed or not set(changed) <= allowed:
        raise ValueError('unexpected mutation extent')
    audit = dict(function=group[0]['function'], first_address=hex(group[0]['address']),
                 permutation=permutation, original=group, changed_bytes=len(changed),
                 original_sha256=hashlib.sha256(blob).hexdigest(),
                 candidate_sha256=hashlib.sha256(result).hexdigest(),
                 unchanged_outside_bundle=True, gpu_validated=False)
    return bytes(result), audit


def emit_header(blob, output):
    pats = ['_Z23kernel_pinning_pipelineILb1ELi0EE', '_Z23kernel_pinning_pipelineILb1ELi2EE',
            '_Z22qsb_root_group_prepare', '_Z22qsb_invert_super_roots',
            '_Z21qsb_root_group_finish', '_Z19kernel_build_gtable', '_Z18qsb_table_offset_y']
    names = list(sections(blob))
    symbols = []
    for p in pats:
        hits = [n[6:] for n in names if n.startswith('.text.'+p)]
        if len(hits) != 1:
            raise ValueError(f'kernel lookup: {p}: {hits}')
        symbols.append(hits[0])
    encoded = base64.b64encode(blob).decode()
    lines = [encoded[i:i+120] for i in range(0, len(encoded), 120)]
    output.write_text('// Generated by sass_order.py; see SASS-ORDER.md and COPYING.\n'
                      '#pragma once\n#include <stddef.h>\n'
                      f'static const size_t qsb_carrier_cubin_bytes = {len(blob)};\n'
                      f'static const char qsb_carrier_cubin_sha256[] = "{hashlib.sha256(blob).hexdigest()}";\n'
                      'static const char *const qsb_carrier_kernel_names[] = {\n'+
                      ''.join(f'"{s}",\n' for s in symbols)+'};\n'+
                      f'static const unsigned qsb_carrier_b64_lines = {len(lines)};\n'+
                      'static const char *const qsb_carrier_b64[] = {\n'+
                      ''.join(f'"{s}",\n' for s in lines)+'};\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cubin', type=Path)
    ap.add_argument('--cuobjdump', default='cuobjdump')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    blob = args.cubin.read_bytes()
    sass = subprocess.check_output([args.cuobjdump, '-sass', str(args.cubin)], text=True)
    candidate, audit = patch(blob, sass)
    args.out.mkdir(exist_ok=True, parents=True)
    emit_header(blob, args.out/'qsb_carrier_control_sm89.h')
    emit_header(candidate, args.out/'qsb_carrier_sm89.h')
    (args.out/'sass-order-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'original'}, indent=2))


if __name__ == '__main__':
    main()
