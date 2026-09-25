#!/usr/bin/env python3
"""Rebuild the generic N=24/sm89 payload with CUDA 12.8.93 (no GPU needed).

Run from anywhere: python3 /path/to/candidates/subset/regenerate_native.py
Outputs only native_image.cuh, native_manifest.cuh and native_manifest.json.
The fixed benchmark command does not need to run this generator.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import struct
import os

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def source_inputs():
    sources = {}
    def visit(path):
        rel = str(path.relative_to(ROOT))
        if rel in sources or path.name in ('native_manifest.cuh', 'native_image.cuh'):
            return
        data = path.read_bytes()
        sources[rel] = sha(data)
        for inc in re.findall(r'^\s*#include\s+"([^"]+)"', data.decode(), re.M):
            visit((path.parent / inc).resolve())
    visit(ROOT / 'subset.cu')
    return sources

def check_contract(ptx):
    contract = json.loads((ROOT / 'native_contract.json').read_text())
    expected = struct.pack('<44I', *contract['descriptor_words'])
    m = re.search(r'\.const\s+\.align\s+4\s+\.b8\s+QSB_S3_DESC\[(\d+)\]\s*=\s*\{([^}]*)\}', ptx)
    if not m or int(m[1]) != len(expected):
        raise ValueError('P18 descriptor size differs from source contract')
    actual = bytes(int(x.strip()) for x in m[2].split(','))
    if len(actual) > len(expected):
        raise ValueError('P18 descriptor initializer is oversized')
    actual += bytes(len(expected)-len(actual))
    if actual != expected:
        raise ValueError('P18 descriptor bytes differ from source contract')
    header = (ROOT / 'native_contract.cuh').read_text()
    for name, values in [('native_contract_desc', contract['descriptor_words']),
                         ('native_contract_banks', [x for row in contract['banks'] for x in row])]:
        m = re.search(name+r'\[\d+\] = \{([^}]*)\}', header)
        if not m or [int(x.strip()) for x in m[1].split(',')] != values:
            raise ValueError('Host contract/header mismatch')
    if contract['terms'] != 11 or contract['total_entries'] != 354501773 or len(contract['banks']) != 8:
        raise ValueError('Unexpected P18 contract dimensions')
    return contract

def verify_only():
    m = json.loads((ROOT / 'native_manifest.json').read_text())
    if m['sources'] != source_inputs():
        raise ValueError('Source changed after native image generation')
    if m['contract'] != json.loads((ROOT / 'native_contract.json').read_text()):
        raise ValueError('Contract changed after native image generation')
    for filename, key in [('native_manifest.cuh','generated_header_sha256'), ('native_image.cuh','generated_image_sha256')]:
        if sha((ROOT / filename).read_bytes()) != m[key]:
            raise ValueError('Generated metadata/image mismatch')
    image = base64.b64decode(''.join(re.findall(r'"([A-Za-z0-9+/=]+)"', (ROOT/'native_image.cuh').read_text())))
    if len(image) != m['cubin_bytes'] or sha(image) != m['cubin_sha256']:
        raise ValueError('Embedded payload differs from manifest')
    print('PASS: exact recursive source, contract, metadata and image identity')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', help='Existing CUDA container with repository mounted at /work')
    parser.add_argument('--verify-only', action='store_true', help='Check source/payload pins without compiling')
    args = parser.parse_args()
    if args.verify_only:
        verify_only()
        return
    before_sources = source_inputs()
    prefix = ['docker', 'exec', '-w', '/work/candidates/subset', args.docker] if args.docker else []
    nvcc_version = subprocess.check_output(prefix + ['nvcc', '--version'], text=True)
    ptxas_version = subprocess.check_output(prefix + ['ptxas', '--version'], text=True)
    for version in (nvcc_version, ptxas_version):
        if 'V12.8.93' not in version:
            raise SystemExit('Reproducible payload requires CUDA V12.8.93')
    with tempfile.TemporaryDirectory(prefix='.qsb-native-', dir=ROOT) as temp:
        ptx = Path(temp) / 'native.ptx'
        cubin = Path(temp) / 'native.cubin'
        subprocess.run(prefix + ['nvcc', '-O3', '-DQSB_ZEROS_N=24', '-DQSB_NATIVE_MODULE=0',
                        '--ptx', 'subset.cu', '-o', str(ptx.relative_to(ROOT))], cwd=ROOT, check=True)
        subprocess.run(prefix + ['ptxas', '-arch=sm_89', '-O3', '-v', str(ptx.relative_to(ROOT)),
                                 '-o', str(cubin.relative_to(ROOT))], cwd=ROOT, check=True)
        text = ptx.read_text()
        payload = cubin.read_bytes()
        contract = check_contract(text)
        if before_sources != source_inputs():
            raise ValueError("Source changed during native generation")
        entries = []
        for symbol, params in re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\(([^)]*)\)', text, re.S):
            types = re.findall(r'\.param\s+\.(u\d+)\s+\w+', params)
            if len(types) != params.count('.param'):
                raise SystemExit('Unsupported PTX parameter type')
            m = re.match(r'_Z(\d+)', symbol)
            name = symbol[m.end():m.end() + int(m[1])]
            entries.append(dict(name=name, symbol=symbol, bytes=[int(t[1:])//8 for t in types]))
        globals_ = []
        for space, align, bits, name, array in re.findall(
            r'^\.(const|global)\s+\.align\s+(\d+)\s+\.([bu]\d+)\s+(\w+)(?:\[(\d+)\])?', text, re.M):
            globals_.append(dict(symbol=name, space=space, alignment=int(align),
                                 bytes=int(bits[1:])//8 * int(array or 1)))
        expected = {'kernel_build_gtable', 'kernel_epoch_groups', 'kernel_build_epochs_inc', 'kernel_build_first_flat', 'kernel_digest', 'kernel_gt_heal_scan'}
        if {e['name'] for e in entries} != expected:
            raise SystemExit(f'Unexpected module shape: {len(entries)} kernels, {len(globals_)} globals')
        digest = sha(payload)
        lines = ['// Generated by regenerate_native.py; do not edit.',
                 '#define QSB_NATIVE_FILE "native_sm89.cubin"',
                 f'#define QSB_NATIVE_BYTES {len(payload)}',
                 'static const unsigned char payload_sha256[32] = {' +
                 ','.join('0x'+digest[i:i+2] for i in range(0,64,2)) + '};',
                 'static Kernel kernels[] = {']
        for e in entries:
            lines.append('{"'+e['name']+'", "'+e['symbol']+'", '+str(len(e['bytes']))+
                         ', {'+','.join(map(str,e['bytes']))+'}, NULL, false},')
        host_text = '\n'.join((ROOT / f).read_text() for f in
            ('tests/gpu_epochs/tree.cu', 'tests/gpu_epochs/window_schedule_shared.cuh'))
        used = set(re.findall(r'QSB_TO_SYMBOL\(\s*(\w+)', host_text))
        used.update(re.findall(r'QSB_FROM_SYMBOL\([^,]+,\s*(\w+)', host_text))
        used.discard('QSB_LANE_CLASS')  # Frozen QSB_950_PACK=0: no active host transfer.
        if len(used) != 16 or not used <= {g['symbol'] for g in globals_}:
            raise SystemExit('Unexpected host symbol routing')
        lines += ['};', 'static Global globals[] = {']
        lines += [f'{{"{g["symbol"]}", {g["bytes"]}, NULL}},' for g in globals_ if g['symbol'] in used]
        lines += ['};']
        # Hash the recursive source inputs, excluding generated metadata. The
        # latter is excluded from the device build by QSB_NATIVE_MODULE=0.
        sources = before_sources
        manifest = dict(toolchain='CUDA 12.8.93', arch='sm_89', zeros_n=24,
                        frontend=['nvcc', '-O3', '-DQSB_ZEROS_N=24', '-DQSB_NATIVE_MODULE=0', '--ptx', 'subset.cu'],
                        assembler=['ptxas', '-arch=sm_89', '-O3'],
                        cubin_bytes=len(payload), cubin_sha256=digest,
                        ptx_sha256=sha(ptx.read_bytes()), entries=entries,
                        globals=globals_, host_globals=sorted(used), sources=sources, contract=contract)
        encoded = base64.b64encode(payload).decode('ascii')
        image = '// Generic CUDA module, base64 encoded; generated by regenerate_native.py.\n'
        image += 'static const char *native_b64[] = {\n'
        image += ''.join('"'+encoded[i:i+120]+'",\n' for i in range(0,len(encoded),120))
        image += '};\n'
        generated_header = '\n'.join(lines)+'\n'
        manifest['generated_header_sha256'] = sha(generated_header.encode())
        manifest['generated_image_sha256'] = sha(image.encode())
        outputs = {'native_image.cuh': image, 'native_manifest.cuh': generated_header,
                   'native_manifest.json': json.dumps(manifest, indent=2)+'\n'}
        # Stage every output; publish the identity-bearing JSON last.
        for name, content in outputs.items():
            (Path(temp)/name).write_text(content)
        for name in outputs:
            os.replace(Path(temp)/name, ROOT/name)
        verify_only()
        print(json.dumps({k:manifest[k] for k in ('cubin_bytes','cubin_sha256','ptx_sha256')}))

if __name__ == '__main__':
    main()
