#!/usr/bin/env python3
"""Compare the source-extracted gate, its four switch combinations and independent hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

SWITCHES = ('QSB_HOST_GATE_JOINT', 'QSB_HOST_RECID_DIGEST_REUSE')
VARIANTS = {'off': (0, 0), 'joint': (1, 0), 'recid': (0, 1), 'on': (1, 1)}
FROZEN = Path(__file__).resolve().parents[3] / 'research/stacking-20260921/frontier94'


def between(text, first, last):
    return text.split(first, 1)[1].split(last, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', type=Path, default=FROZEN / 'astra-finalization/parent')
    ap.add_argument('--problem', type=Path, default=FROZEN / 'astra-recovery/parent/problems')
    args = ap.parse_args()
    work = Path(os.environ['QSB_VERIFY_WORK'])
    work.mkdir(parents=True, exist_ok=True)
    base = Path(os.environ['QSB_BASE_DIR']) / 'pinning.cu'
    candidate = Path(os.environ['QSB_ARM_DIR']) / 'pinning.cu'
    sys.path.insert(0, str(args.control / 'harness'))
    import crypto as crypto
    import problem as problem
    prob = json.loads((args.problem / 'pinning.json').read_text())
    rng = random.Random(3105)
    pairs = [(s, l) for s in (0, 0x80000000, 0xfffffffe, 0xffffffff)
             for l in (0, 499999999, 500000000, 0xffffffff)]
    pairs += [(rng.randrange(0x80000000, 0xffffffff), rng.randrange(500000000, 0xffffffff))
              for _ in range(1008)]
    samples = [(s, l, r) for s, l in pairs for r in (0, 1)]
    sample_file = work / 'samples.txt'
    sample_file.write_text(''.join(f'{s} {l} {r}\n' for s, l, r in samples))
    includes = '''#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
#define QSB_ZEROS_N 8
static unsigned char captured_pub[33], captured_hash[32];
static int digest_calls;
static unsigned char *capture_sha(const unsigned char *p, size_t n, unsigned char *out) {
    auto result = SHA256(p,n,out);
    if(n==32) ++digest_calls;
    if(n==33) { memcpy(captured_pub,p,33); memcpy(captured_hash,out,32); }
    return result;
}
#define SHA256 capture_sha
'''
    footer = r'''
    FILE *samples=fopen(argv[2],"r"); assert(samples);
    unsigned seq,lt; int recid;
    while(fscanf(samples,"%u %u %d",&seq,&lt,&recid)==3) {
        memset(captured_pub,0,sizeof(captured_pub)); memset(captured_hash,0,sizeof(captured_hash));
        int hit=qsb_host_exact_hit(&pp,seq,lt,recid,gate_grp,gate_ctx,gate_order,gate_nri,gate_R);
        printf("ROW %u %u %d %d ",seq,lt,recid,hit);
        for(auto b:captured_pub) printf("%02x",b);
        printf(" "); for(auto b:captured_hash) printf("%02x",b);
        digest_calls=0;
        int accepted=qsb_gate_accept(&pp,seq,lt,recid,gate_grp,gate_ctx,gate_order,gate_nri,gate_R);
        printf(" %d\n",accepted);
#if QSB_HOST_RECID_DIGEST_REUSE
        assert(digest_calls==1);
#else
        assert(digest_calls==(hit ? 1 : 2));
#endif
    }
    pp.suffix_len=120;
    assert(qsb_host_exact_hit(&pp,0,0,0,gate_grp,gate_ctx,gate_order,gate_nri,gate_R)==0);
    assert(qsb_gate_accept(&pp,0,0,0,gate_grp,gate_ctx,gate_order,gate_nri,gate_R)==-1);
    pp.suffix_len=75; pp.seq_offset=73;
    assert(qsb_host_exact_hit(&pp,0,0,0,gate_grp,gate_ctx,gate_order,gate_nri,gate_R)==0);
    pp.seq_offset=31; pp.lt_offset=73;
    assert(qsb_host_exact_hit(&pp,0,0,0,gate_grp,gate_ctx,gate_order,gate_nri,gate_R)==0);
    fclose(samples); free(pp.suffix);
    EC_POINT_free(gate_R); BN_free(gate_ry); BN_free(gate_rx); BN_free(gate_nri); BN_free(gate_order);
    BN_CTX_free(gate_ctx); EC_GROUP_free(gate_grp);
}
'''
    outputs, binaries = {}, {}
    builds = [('base', base, None), ('default', candidate, None)]
    builds += [(label, candidate, values) for label, values in VARIANTS.items()]
    for label, source, values in builds:
        text = source.read_text()
        switch = ''
        if '#ifndef QSB_HOST_GATE_JOINT' in text:
            switch = '#ifndef QSB_HOST_GATE_JOINT' + between(text, '#ifndef QSB_HOST_GATE_JOINT', '#ifndef QSB_FEED_BLOCK')
        params = between(text, '/* Params loader for pinning2.bin */', 'static int load_pinning2')
        start = text.index('static int load_pinning2')
        end = text.index('\n}', start) + 2
        loader = text[start:end]
        start = text.index('static int qsb_host_zeros')
        accept = text.index('static int qsb_gate_accept', start)
        end = text.index('\n}', accept) + 2
        gate = text[start:end]
        start = text.index('    EC_GROUP *gate_grp = EC_GROUP_new_by_curve_name')
        end = text.index('\n#endif\n#if QSB_CPU_GRIND', start)
        setup = text[start:end]
        program = includes + switch + params + loader + '\n' + gate + '\n#line 1 "host_joint_oracle_main"\nint main(int argc,char **argv) {\n'
        program += '    assert(argc==3); pinning2_params_t pp; assert(load_pinning2(argv[1],&pp)==0);\n'
        program += setup + '\n#line 100 "host_joint_oracle_main"\n' + footer
        cpp = work / 'gate.cpp'
        cpp.write_text(program)
        exe = work / (label + '.bin')
        command = ['g++', '-O2', '-w', '-std=c++17', str(cpp), '-lcrypto', '-o', str(exe)]
        if values is not None:
            command[1:1] = [f'-D{name}={value}' for name, value in zip(SWITCHES, values)]
        subprocess.run(command, check=True)
        binaries[label] = hashlib.sha256(exe.read_bytes()).hexdigest()
        result = subprocess.run([str(exe), str(args.problem / 'pinning.bin'), str(sample_file)],
                                check=True, capture_output=True, text=True)
        (work / (label + '.txt')).write_text(result.stdout)
        outputs[label] = [line.split() for line in result.stdout.splitlines() if line.startswith('ROW ')]
        assert len(outputs[label]) == len(samples), (label, len(outputs[label]))
    assert binaries['base'] == binaries['off'], ('host OFF binary differs', binaries)
    for label, rows in outputs.items():
        assert rows == outputs['base'], ('host gate output differs', label)
    for label in ('joint', 'recid', 'on'):
        assert binaries[label] != binaries['off'], (label + ' build did not change the host executable')
    reference_checks = 0
    for i in list(range(32)) + list(range(32, len(samples), 67)):
        seq, lt, recid = samples[i]
        expected = problem.candidate_hash(prob, {'sequence': seq, 'locktime': lt}, recid)
        actual = bytes.fromhex(outputs['on'][i][6])
        assert actual == expected, ('independent hash mismatch', seq, lt, recid)
        reference_checks += 1
    positives = [0, 0]
    fallback = 0
    for i, row in enumerate(outputs['on']):
        hit, accepted = int(row[4]), int(row[7])
        recid = samples[i][2]
        other_hit = int(outputs['on'][i ^ 1][4])
        expected = recid if hit else 1 - recid if other_hit else -1
        assert accepted == expected, ('fallback mismatch', i, accepted, expected)
        assert hit == int(crypto.leading_zero_bits(bytes.fromhex(row[6])) >= 8)
        positives[recid] += hit
        fallback += int(not hit and other_hit)
    assert min(positives) > 0 and fallback > 0, (positives, fallback)
    receipt = {'samples': len(samples), 'independent_full_hash_checks': reference_checks,
               'positive_recids': positives, 'fallback_accepts': fallback, 'mismatches': 0,
               'host_binary_sha256': binaries, 'host_off_byte_identical': True,
               'on_host_code_changed': True, 'source_sha256': {
                   'parent': hashlib.sha256(base.read_bytes()).hexdigest(),
                   'candidate': hashlib.sha256(candidate.read_bytes()).hexdigest()},
               'scope': 'source-extracted production gate under all joint/digest switches; full pubkey/hash equality, independent Python crypto witnesses, digest-call counts, invalid geometry; no GPU execution'}
    (work / 'host-oracle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('ORACLE_IDENTICAL ' + json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
