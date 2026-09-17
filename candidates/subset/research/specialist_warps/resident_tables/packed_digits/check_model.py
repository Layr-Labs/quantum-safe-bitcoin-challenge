#!/usr/bin/env python3
"""Independent Python integer/limb model; does not execute CUDA or C++ helpers."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK64 = (1 << 64) - 1
LIMIT = 1 << 256

def sha(data):
    return hashlib.sha256(data).hexdigest()

def limbs(x):
    return [(x >> (64 * j)) & MASK64 for j in range(4)]

def integer(xs):
    return sum(x << (64 * j) for j, x in enumerate(xs))

def subtract(xs, ys):
    out, borrow = [], 0
    for x, y in zip(xs, ys):
        v = x - y - borrow
        out.append(v & MASK64)
        borrow = int(v < 0)
    return out, borrow

def setup_limb_model(k):
    """Mirror actual unsigned carry/select setup, compare independent modulo model."""
    a, order = limbs(k), limbs(N)
    reduced, borrow = subtract(a, order)
    a = a if borrow else reduced
    t = [(a[j] << 1 | (a[j-1] >> 63 if j else 0)) & MASK64 for j in range(4)]
    d, borrow = subtract(t, order)
    m = d if (a[3] >> 63) | (1 - borrow) else t
    complement, _ = subtract(order, m)
    odd = m[0] & 1
    return integer(m if odd else complement), 1 - odd

def setup_reference(k):
    m = (2 * (k % N)) % N
    return (m, 0) if m & 1 else (N - m, 1)

def encode_field(f, sign, last=False):
    t = f >> 15
    idx = f & 32767 if last else (f ^ ((t - 1) & 0xffffffff)) & 32767
    neg = (0 if last else t ^ 1) ^ sign
    return idx | (neg << 15)

def direct(M, sign):
    return [encode_field((M >> (16*c+1)) & 65535, sign, c == 15) for c in range(16)]

def value(p):
    return (1 - 2 * (p >> 15)) * (2 * (p & 32767) + 1)

def serial_reference(M, sign):
    out = []
    for _ in range(15):
        e = (M & 131071) - 65536
        out.append((1 - 2 * sign) * e)
        M = (M - e) >> 16
    out.append((1 - 2 * sign) * M)
    return out

def pack_stream(M, sign, reverse=False, omit_cross=False, wrong_last=False):
    xs = limbs(M)
    for k in (range(3, -1, -1) if reverse else range(4)):
        lo, hi = xs[k], (xs[k+1] & 1) if k < 3 else 0
        packed = 0
        for j in range(4):
            f = (lo >> (16*j+1)) & 65535
            if j == 3 and not omit_cross:
                f |= hi << 15
            p = encode_field(f, sign, k == 3 and j == 3 and not wrong_last)
            packed |= p << (16*j)
        xs[k] = packed
    return xs

def unpack(xs):
    return [(xs[c >> 2] >> (16 * (c & 3))) & 65535 for c in range(16)]

def inverse_digits(ps):
    sign = ps[15] >> 15
    fs = []
    for c, p in enumerate(ps):
        idx, neg = p & 32767, p >> 15
        if c == 15:
            f = idx
        else:
            t = 1 ^ neg ^ sign
            f = (idx if t else idx ^ 32767) | (t << 15)
        fs.append(f)
    return 1 + 2 * sum(f << (16*c) for c, f in enumerate(fs)), sign

def closure(root):
    seen = set()
    def visit(p):
        p = p.resolve()
        assert p.is_relative_to(root) and p.is_file()
        if p in seen:
            return
        seen.add(p)
        for rel in re.findall(r'^\s*#include\s+"([^"]+)"', p.read_text(), re.M):
            visit(p.parent / rel)
    visit(root / 'subset.cu')
    visit(root / 'tests/gpu_epochs/tree_audit.cu')
    hashes = {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(seen)}
    return hashes, sha(json.dumps(hashes, sort_keys=True).encode())

def body(text, name):
    begin = text.index(name + '(') if name + '(' in text else text.index(name + ' (')
    opening = text.index('{', begin)
    level = 1
    end = opening + 1
    while level:
        level += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[begin:end]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE.parent/'small32/candidate')
    parser.add_argument('--output', type=Path, default=HERE/'design.json')
    args = parser.parse_args()
    root = args.source.resolve()
    hashes, fingerprint = closure(root)
    assert fingerprint == '008f904cfb385d030c15324690bbbc5d662777c4547d8db5442b3c1b509dedfd'
    device = (root/'tests/gpu_epochs/compact_table_device.cuh').read_text()
    tree = (root/'tests/gpu_epochs/tree.cu').read_text()
    bodies = {name: sha(body(text, name).encode()) for name, text in [
        ('gt_recode_setup', tree), ('gt_direct_digit', device),
        ('qsb_ec192_shared_field_bits', device), ('qsb_ec192_shared_direct_digit', device),
        ('compact_ec192_fixed_xyzz_shared', device), ('kernel_digest_specialist', tree)]}
    rng = random.Random(0x16D1616)
    cases = {0, 1, 2, N-2, N-1, N, N+1, N+2, LIMIT-1}
    for center in [N, N//2, (N+1)//2, LIMIT//2] + [1 << b for b in range(256)]:
        cases.update(x for x in range(center-3, center+4) if 0 <= x < LIMIT)
    exception_path = root.parent/'field-check/chain_recovery/exception-scalars.json'
    witnesses = json.loads(exception_path.read_text())
    cases.update(int(v, 0) for v in witnesses)
    cases.update(rng.getrandbits(256) for _ in range(12000))
    caught = {}
    signs = [0, 0]
    for k in sorted(cases):
        M, sign = setup_limb_model(k)
        assert (M, sign) == setup_reference(k)
        assert M & 1 and 1 <= M <= N
        signs[sign] += 1
        ps, packed = direct(M, sign), pack_stream(M, sign)
        assert unpack(packed) == ps
        assert inverse_digits(ps) == (M, sign)
        es = [value(p) for p in ps]
        assert es == serial_reference(M, sign)
        total = sum(e << (16*c) for c, e in enumerate(es))
        assert total == (1-2*sign)*M and (total - 2*k) % N == 0
        # Existing SHA big-endian state conversion and proposed ring wire ABI.
        digest = [(k >> (32*(7-j))) & 0xffffffff for j in range(8)]
        original_limbs = [(digest[6-2*j] << 32) | digest[7-2*j] for j in range(4)]
        assert integer(original_limbs) == k
        wire = [(packed[j//2] >> (32*(j&1))) & 0xffffffff for j in range(8)]
        received = [wire[2*j] | (wire[2*j+1] << 32) for j in range(4)]
        assert unpack(received) == ps
        mutations = {
            'descending_in_place': unpack(pack_stream(M, sign, reverse=True)),
            'missing_limb_cross_bit': unpack(pack_stream(M, sign, omit_cross=True)),
            'top_digit_nonlast': unpack(pack_stream(M, sign, wrong_last=True)),
            'drop_global_sign': direct(M, 0),
            'old_SHA_word_order_on_packed_wire': unpack([(wire[6-2*j]<<32)|wire[7-2*j] for j in range(4)]),
            'unmasked_cell_extraction': [(packed[c >> 2] >> (16 * (c & 3))) for c in range(16)],
        }
        for name, mutated in mutations.items():
            if mutated != ps and name not in caught:
                c = next(c for c in range(16) if mutated[c] != ps[c])
                caught[name] = {'input': hex(k), 'M': hex(M), 'sign': sign, 'first_window': c,
                                'expected_cell': hex(ps[c]), 'mutated_cell': hex(mutated[c])}
    assert len(caught) == 6
    # Exhaust every nonlast field/sign and top field/sign. Encoding is bijective
    # for a fixed sign on each nonlast cell; the final cell carries global sign.
    exhaustive = 0
    for sign in range(2):
        seen = set()
        for f in range(65536):
            p = encode_field(f, sign)
            assert value(p) == (1-2*sign)*(2*f-65535)
            seen.add(p)
            exhaustive += 1
        assert len(seen) == 65536
        for f in range(32768):
            p = encode_field(f, sign, True)
            assert value(p) == (1-2*sign)*(2*f+1)
            exhaustive += 1
    # Representation bijection covers even invalid setup states independently.
    for _ in range(10000):
        raw = rng.getrandbits(256)
        ps = unpack(limbs(raw))
        M, sign = inverse_digits(ps)
        assert 1 <= M < LIMIT and M & 1 and direct(M, sign) == ps
        assert integer(pack_stream(M, sign)) == raw
    assert direct(*setup_reference(0)) == direct(*setup_reference(N))
    report = {
        'status': 'PASS', 'scope': 'Independent Python arithmetic and unsigned limb models; no extracted C++/CUDA execution',
        'source_fingerprint': fingerprint, 'source_sha256': hashes, 'helper_body_sha256': bodies,
        'checker_sha256': sha(Path(__file__).read_bytes()),
        'input_cases': len(cases), 'global_sign_counts': signs, 'exception_fixture_count': len(witnesses),
        'exception_fixture_sha256': sha(exception_path.read_bytes()),
        'exhaustive_field_sign_cases': exhaustive, 'arbitrary_packed_roundtrips': 10000,
        'negative_controls': caught,
        'wire_protocol': 'word[2*j]=low32(packed_limb[j]); word[2*j+1]=high32(packed_limb[j]); cell c uses bits16*(c%4)..+15 of limb[c/4]',
        'counts': {'ring_payload_bytes': 32, 'ring_read_plus_write_bytes': 64,
                   'arena_scalar_store_bytes': 32, 'old_source_scalar_read_bytes': 224,
                   'new_source_scalar_read_bytes': 128, 'source_read_saving_bytes': 96,
                   'old_native_loop_scalar_read_operand_bytes': 148, 'new_model_loop_scalar_read_bytes': 104,
                   'old_native_loop_global_sign_LDL_bytes': 52,
                   'other_native_loop_local_bytes_no_guaranteed_saving': 312,
                   'setup_calls_moved_consumer_to_producer': 1, 'digit_encodings_moved': 16,
                   'additional_pack_insertions': 16, 'consumer_cell_extractions': 16},
        'recommendation': 'Proceed to isolated prototype with ascending streaming packing and explicit changed wire ABI; native and actual-source control/curve qualification remain required.',
        'limitations': ['No candidate source changed or compiled', 'No speed or spill-removal claim',
                       'Native loop operand count uses prior source-bound SASS review; new native code is unknown',
                       'Raw uint256 hash is not injective after reduction modulo n; only normalized scalar states and odd-M/sign representation have the stated bijections'],
    }
    actual_report = HERE/'digits-results/results.json'
    if actual_report.is_file():
        actual = json.loads(actual_report.read_text())
        report['implementation_followup'] = {
            'source_fingerprint': actual['source_fingerprint'], 'status': actual['status'],
            'report': 'digits-results/results.json', 'report_sha256': sha(actual_report.read_bytes()),
            'scope': actual['scope']}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: report[k] for k in ('status','source_fingerprint','input_cases','exhaustive_field_sign_cases','arbitrary_packed_roundtrips')}))

if __name__ == '__main__':
    main()
