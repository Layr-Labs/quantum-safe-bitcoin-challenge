"""Generate and verify the paired-field and exact recovery native fixtures.

With no binaries or output files this only generates deterministic CPU inputs.
CUDA execution on the shared Pod must run inside its installed GPU wrapper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess

from audit_paired_upper_fold import P, boundary_pairs


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--field-binary', type=Path)
    parser.add_argument('--recovery-binary', type=Path)
    parser.add_argument('--field-output', type=Path)
    parser.add_argument('--recovery-output', type=Path)
    args = parser.parse_args()
    assert bool(args.field_binary) == bool(args.recovery_binary)
    assert bool(args.field_output) == bool(args.recovery_output)
    assert not (args.field_binary and args.field_output)
    args.directory.mkdir(exist_ok=False)
    pairs = boundary_pairs()
    raw_field = struct.pack('<I', len(pairs)) + b''.join(
        a.to_bytes(32, 'little')+b.to_bytes(32, 'little') for a, b in pairs)
    edge = [0, 1, 2, 3, (1 << 32)+977, (P-1)//2, P//2, P//2+1, P-3, P-2, P-1]
    cases = [(v, u, 1, 1, a, b, c) for u in edge for v in edge
             for a, b, c in [(0, 1, 0), (P-1, 1, P-1), (1, P-1, P//2),
                             (P-1, (1 << 192)-1, P-1), (1, 1 << 192, 0)]]
    rng = random.Random(0x51534253554d)
    cases += [tuple(rng.randrange(P) for _ in range(7)) for _ in range(16384)]
    assert len(cases) == 16989
    raw_recovery = b''.join(x.to_bytes(32, 'little') for row in cases for x in row)
    inputs = {'field': raw_field, 'recovery': raw_recovery}
    for name, data in inputs.items():
        (args.directory/(name+'-input.bin')).write_bytes(data)
    proof = {'inputs_generated': True, 'field_pairs': len(pairs), 'recovery_cases': len(cases),
             'input_sha256': {name: sha(data) for name, data in inputs.items()},
             'native_outputs_verified': False}
    if args.field_binary:
        for name, binary in [('field', args.field_binary), ('recovery', args.recovery_binary)]:
            output = args.directory/(name+'-output.bin')
            subprocess.run([str(binary.resolve()), str(args.directory/(name+'-input.bin')),
                            str(output)], check=True, timeout=180)
            setattr(args, name+'_output', output)
            proof[name+'_binary_sha256'] = sha(binary.read_bytes())
    if args.field_output:
        field = args.field_output.read_bytes()
        recovery = args.recovery_output.read_bytes()
        assert len(field) == 96*len(pairs) and len(recovery) == 144*len(cases)
        for i, (a, b) in enumerate(pairs):
            values = [int.from_bytes(field[96*i+32*j:96*i+32*(j+1)], 'little') for j in range(3)]
            assert values[0] == values[1] == values[2] and values[0] % P == a*b % P, i
        for i, row in enumerate(cases):
            vb, tb, ri, wi, a, b, c = row
            u, v = tb*wi % P, vb*ri % P
            l, m, s = (u-v) % P, (u+v) % P, 2*u % P
            x1, x2 = (s*(l-c)+a) % P, (s*(m-c)+a) % P
            flags = ((l*(a-x1)-b) % P & 1) | (((b-m*(a-x2)) % P & 1) << 1)
            for offset in (144*i, 144*i+72):
                got = (int.from_bytes(recovery[offset:offset+32], 'little'),
                       int.from_bytes(recovery[offset+32:offset+64], 'little'),
                       int.from_bytes(recovery[offset+64:offset+72], 'little'))
                assert got == (x1, x2, flags), (i, offset, got)
        proof.update(passed=True, native_outputs_verified=True, field_outputs=len(pairs)*3,
                     recovery_implementations=2,
                     output_sha256={'field': sha(field), 'recovery': sha(recovery)})
    (args.directory/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
