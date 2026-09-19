"""Independent modular oracle for the shipped square/multiply native audit.

This launches CUDA: on the shared Pod run the entire command through
bash /workspace/shared/run-gpu pinning-b python3 audit_final_fold.py BINARY OUTPUT_DIR
It does not modify the organizer benchmark or produce a ranked score.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('output_directory', type=Path)
args = parser.parse_args()
binary = args.binary.resolve()
root = args.output_directory.resolve()
root.mkdir(exist_ok=False)
B = 1 << 256
K = (1 << 32) + 977
P = B-K
edge = [0, 1, 2, 977, K-1, K, (1 << 64)-1, 1 << 64,
        B//2, P-65538, P-65537, P-65536, P-2, P-1, P, B-1]
pairs = list(itertools.product(edge, repeat=2))
rng = random.Random(202609190313)
pairs += [(rng.randrange(B), rng.randrange(B)) for _ in range(4096)]
# Deliberately cover a neighbourhood of the known final-fold counterexample.
pairs += [(P-d, P-e) for d in (65535, 65536, 65537, 65538, 65539)
          for e in (65535, 65536, 65537, 65538, 65539)]
raw = struct.pack('<I', len(pairs)) + b''.join(
    a.to_bytes(32, 'little')+b.to_bytes(32, 'little') for a, b in pairs)
(root/'inputs.bin').write_bytes(raw)
subprocess.run([str(binary), str(root/'inputs.bin'), str(root/'outputs.bin')], check=True)
output = (root/'outputs.bin').read_bytes()
assert len(output) == len(pairs)*160
for i, (a, b) in enumerate(pairs):
    values = [int.from_bytes(output[160*i+32*j:160*i+32*(j+1)], 'little') for j in range(5)]
    assert values[0] == values[1] == values[2], (i, 'multiply alias')
    assert values[3] == values[4], (i, 'square alias')
    expected = [a*b % P]*3 + [a*a % P]*2
    assert all(x % P == y for x, y in zip(values, expected)), (i, a, b, values, expected)
result = {
    'passed': True, 'pairs': len(pairs), 'outputs_verified': 5*len(pairs),
    'scope': 'Native arithmetic and alias diagnostic; not a full benchmark or score.',
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'input_sha256': hashlib.sha256(raw).hexdigest(),
    'output_sha256': hashlib.sha256(output).hexdigest(),
}
(root/'proof.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
