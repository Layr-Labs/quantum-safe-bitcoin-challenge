"""Reproduce the table guard's actual CUDA edge-case validation.

Run on an available CUDA GPU; shared machines must use their normal GPU lease.
The generated inputs and Python integer oracle are independent of benchmark
instances. --emit-input creates only the deterministic fixture, without CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def fixture():
    word = 1 << 64
    low_p = word - (1 << 32) - 977
    lows = [0, 1, 2, (1 << 32)-1, 1 << 32, (1 << 63)-1, 1 << 63,
            low_p-2, low_p-1, low_p, low_p+1, low_p+2, word-2, word-1]
    highs = [0, 1, (1 << 64)-1, 1 << 64, (1 << 128)-1, 1 << 128, (1 << 192)-1]
    rng = random.Random(61082467)
    highs += [rng.getrandbits(192) for _ in range(80)]
    ys = [(h << 64) | low for h in highs for low in lows]
    ys += [rng.getrandbits(256) for _ in range(6000)]
    return b''.join(struct.pack('<5Q', *((y >> (64*i)) & (word-1) for i in range(4)), mask)
                    for y in ys for mask in (0, word-1))


def verify(data, result, status):
    assert status['cuda_completed'] and status['guard_scan_passed']
    assert len(data) % 40 == 0 and len(result) == len(data)//40*128
    word = 1 << 64
    field = (1 << 256) - (1 << 32) - 977
    safe = fallback = negative_fallback = 0
    for i, (record, output) in enumerate(zip(struct.iter_unpack('<5Q', data),
                                           struct.iter_unpack('<16Q', result))):
        y = sum(record[j] << (64*j) for j in range(4))
        mask = record[4]
        assert mask in (0, word-1)
        old = sum(output[4+j] << (64*j) for j in range(4))
        fast = sum(output[12+j] << (64*j) for j in range(4))
        expected = y if mask == 0 else (field-y) % (1 << 256)
        assert old == expected, ('full carry', i)
        x = tuple(0x123456789abcdef0+i*17+j for j in range(4))
        assert output[:4] == output[8:12] == x
        condition = record[0] <= field % word
        assert (fast if condition else old) == expected, ('selected', i)
        if condition:
            assert fast == old
            safe += 1
        else:
            fallback += 1
            if mask:
                assert fast != old
                negative_fallback += 1
    assert safe+fallback == status['cases'] == 14436
    assert fallback == status['fallback'] == 696 and negative_fallback == 348
    return {'passed': True, 'cases': safe+fallback, 'safe': safe,
            'fallback': fallback, 'required_negative_fallback': negative_fallback,
            'actual_host_scan_both_outcomes': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--emit-input', type=Path)
    args = parser.parse_args()
    data = fixture()
    assert hashlib.sha256(data).hexdigest() == '01906c28bab76f7c617beeed7ff46b0dbb653a8b7df8141e5d84a8bf9b223f9e'
    if args.emit_input:
        args.emit_input.write_bytes(data)
        print(json.dumps({'cases': len(data)//40, 'cuda_executed': False,
                          'input_sha256': hashlib.sha256(data).hexdigest()}))
        return
    source = args.source.resolve()
    with tempfile.TemporaryDirectory(prefix='qsb-table-guard-') as temp:
        work = Path(temp)
        (work/'input.bin').write_bytes(data)
        subprocess.run([args.nvcc, '-O3', '-DQSB_ZEROS_N=24', '-I'+str(source),
                        str(Path(__file__).with_suffix('.cu')), '-o', str(work/'audit'),
                        '-lcrypto', '-lm'], check=True)
        status = json.loads(subprocess.check_output([
            str(work/'audit'), str(work/'input.bin'), str(work/'output.bin'),
        ], text=True, timeout=60))
        output = (work/'output.bin').read_bytes()
        proof = verify(data, output, status)
        proof.update(input_sha256=hashlib.sha256(data).hexdigest(),
                     output_sha256=hashlib.sha256(output).hexdigest(),
                     source_sha256=hashlib.sha256((source/'pinning.cu').read_bytes()).hexdigest(),
                     verifier='Independent Python integer oracle for actual CUDA outputs')
        print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
