"""Reproduce the raw field fixture and verify native add/sub alias outputs.

Fixture generation is CPU-only. Executing a CUDA binary on the shared research
Pod requires the installed run-gpu lease. This is not a throughput benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from audit_small_ptx import boundary_pairs
from verify_small_field_native import verify


def fixture():
    pairs = boundary_pairs()
    assert len(pairs) == 8678
    return struct.pack('<I', len(pairs)) + b''.join(
        a.to_bytes(32, 'little') + b.to_bytes(32, 'little') for a, b in pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    if not args.inputs_only and args.binary is None:
        parser.error('supply --binary or --inputs-only')
    root = args.output_directory.resolve()
    root.mkdir(exist_ok=False)
    inputs, outputs = root / 'inputs.bin', root / 'outputs.bin'
    inputs.write_bytes(fixture())
    if args.inputs_only:
        print(json.dumps({'pairs': 8678, 'cuda_executed': False,
                          'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest()}))
        return
    binary = args.binary.resolve()
    subprocess.run([str(binary), str(inputs), str(outputs)], check=True)
    result = verify(inputs, outputs)
    result['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (root / 'proof.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['passed'] and result['outputs'] == 78102
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
