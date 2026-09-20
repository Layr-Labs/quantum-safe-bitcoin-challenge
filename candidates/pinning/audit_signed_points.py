"""Reproduce final-window scalar cases for the public QSB pinning benchmark.

--inputs-only performs no CUDA work. Otherwise both supplied audit binaries
execute CUDA and must run inside the shared run-gpu lease on the research Pod.
The native binaries independently compare coordinates with OpenSSL.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess


def fixture():
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    b = 1 << 256
    t = ((1 << 17) - 1) << 239
    delta = n - t
    values = {0, 1, 2, n - 1, n, n + 1, b - 1, delta, t}
    for k in (delta, t):
        for change in (-65537, -2, -1, 0, 1, 2, 65537):
            values.add(k + change)
    inv2 = pow(2, -1, n)
    lam = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
    for chunk in range(15):
        width = 18 if chunk == 0 else 17
        shift = 0 if chunk == 0 else 17 * chunk + 1
        for index in (0, 1, 123, (1 << (width - 1)) - 1):
            for factor in (1, 2, lam, 2 * lam):
                for sign in (-1, 1):
                    values.add(sign * ((2 * index + 1) << shift) * inv2 * factor % n)
    rng = random.Random(202609191138)
    values = sorted(values) + [rng.randrange(b) for _ in range(256)]
    raw = struct.pack('<I', len(values))
    raw += b''.join(k.to_bytes(32, 'little') for k in values)
    assert len(values) == 753
    assert hashlib.sha256(raw).hexdigest() == 'eb7bfed01e0c657b04225b1002be9f38f4f20e49a86eff349c8e493bc023f24b'
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--guarded', type=Path)
    parser.add_argument('--full', type=Path)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    if not args.inputs_only and (args.guarded is None or args.full is None):
        parser.error('supply both audit binaries, or use --inputs-only')
    root = args.output_directory.resolve()
    root.mkdir(exist_ok=False)
    inputs = root / 'inputs.bin'
    inputs.write_bytes(fixture())
    if args.inputs_only:
        print(json.dumps({'cases': 753, 'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(), 'cuda_executed': False}))
        return
    results = []
    for mode, binary in (('guarded', args.guarded.resolve()), ('full', args.full.resolve())):
        for base in ('G', 'dense'):
            proof_path = root / (mode + '-' + base + '.json')
            command = [str(binary), str(inputs), str(proof_path)]
            if base == 'dense':
                command.append('dense')
            subprocess.run(command, check=True)
            proof = json.loads(proof_path.read_text())
            assert proof['passed'] and proof['native_points'] == 753
            assert proof['dense_base'] == (base == 'dense')
            coordinates = Path(str(proof_path) + '.points.bin')
            assert coordinates.stat().st_size == 72 * 753
            results.append({'mode': mode, 'base': base, 'proof': proof,
                            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                            'output_sha256': hashlib.sha256(coordinates.read_bytes()).hexdigest()})
    summary = {'passed': True, 'native_points': 3012, 'results': results,
               'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
               'scope': 'Native scalar-point diagnostic with OpenSSL coordinate oracle; not a production benchmark or score.'}
    (root / 'proof.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
