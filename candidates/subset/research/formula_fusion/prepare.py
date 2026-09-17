#!/usr/bin/env python3
"""Isolated cubic-identity recovery on the frozen fused small48 source."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity
from check_candidate import function

BASE = ROOT / 'research/asymmetric_windows/small48/frontier_fused/candidate'
DEST = HERE / 'candidate'
EXPECTED = 'b832d7ba4ec2b2554aa5279e46858a5163e83e3ad7e714e9b108551165c95f7b'
assert not DEST.exists(), 'Preserve immutable prepared candidate'
base_id = source_identity(BASE)
assert base_id['source_fingerprint'] == EXPECTED
files = {p: (BASE/p).read_text() for p in [*base_id['source_sha256'], 'COPYING']}
path = 'tests/gpu_epochs/tree.cu'; tree = files[path]
declaration = '__device__ __constant__ uint64_t QSB_U2R[8];'
assert tree.count(declaration) == 1
tree = tree.replace(declaration, declaration + '\n// Runtime-problem constant:3*xR^2 mod p; never a precomputed problem answer.\n__device__ __constant__ uint64_t QSB_U2R_C3X2[4];')

prepare = r'''__device__ __forceinline__ void qsb_xyzz_finish_prepare(
    uint64_t *X_D, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *xR, uint64_t *W
) {
    uint64_t t[4];
    _ModMult(t, xR, ZZ);
    _ModSub256(t, t, X_D);
    Load256(X_D, t);             // d = A*(xR-xP)
    _ModMult(W, ZZZ, X_D);       // W = B*d; A=ZZ, B=ZZZ
    W[4] = 0;
}'''
finish = r'''__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(
    uint64_t *A, uint64_t *Y, uint64_t *W, uint64_t *ZZZ,
    uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t c3x2[4]={QSB_U2R_C3X2[0],QSB_U2R_C3X2[1],QSB_U2R_C3X2[2],QSB_U2R_C3X2[3]};
    uint64_t m[4],t[4],s[4];
    _ModMult(A, inv);             // h = A/(B*d)
    _ModMult(ZZZ, A);             // k = A/d = 1/(xR-xP)
    _ModMult(Y, A);               // beta = yP*k
    _ModMult(A, yR, ZZZ);         // alpha = yR*k; h dies
    _ModMult(ZZZ, c3x2);          // 3*xR^2*k; k dies
    _ModSqr(W, A);
    _ModAdd256(W, W, W);
    _ModSub256(W, ZZZ);
    _ModAdd256(W, W, xR);         // center = 2*alpha^2 - 3*xR^2*k + xR
    _ModMult(m, A, Y);
    _ModAdd256(m, m, m);          // difference = 2*alpha*beta
    _ModSub256(x1, W, m);
    _ModAdd256(x2, W, m);

    _ModSub256(m, A, Y);          // lambda1 = alpha-beta
    _ModSub256(t, xR, x1);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    uint32_t parities = (uint32_t)(s[0] & 1ULL);
    _ModAdd256(m, A, Y);          // -lambda2 = alpha+beta
    _ModSub256(t, xR, x2);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    parities |= (uint32_t)(((s[0] & 1ULL) ^ 1ULL) << 1);
    return parities;
}'''
old_prepare = function(tree, '__device__ __forceinline__ void qsb_xyzz_finish_prepare(')
old_finish = function(tree, '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')
tree = tree.replace(old_prepare, prepare).replace(old_finish, finish)
comment_start = tree.index('/* XYZZ shared-denominator recovery (transplanted')
prepare_start = tree.index('__device__ __forceinline__ void qsb_xyzz_finish_prepare(', comment_start)
tree = tree[:comment_start] + '''/* Cubic-identity pair recovery. For A=ZZ,B=ZZZ,d=xR*A-X, invert B*d.
 * The finish uses yP^2=xP^3+7 and yR^2=xR^3+7 to eliminate three squares.
 * Requires valid on-curve points and nonzero A,B,d; the caller preserves the
 * original identity-padding/skip behavior for zero denominators. */
''' + tree[prepare_start:]
comment_start = tree.index('/* Stage 2. C=ZZ*d^2')
finish_start = tree.index('__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(', comment_start)
tree = tree[:comment_start] + '''/* k=1/(xR-xP),alpha=yR*k,beta=yP*k.
 * x1,2 = 2*alpha^2 - 3*xR^2*k + xR -/+ 2*alpha*beta.
 * The same slopes alpha-/+beta recover the two compressed-key parities.
 * Existing A,Y,B,W storage is reused; the inverse tree itself is unchanged. */
''' + tree[finish_start:]
call = 'qsb_xyzz_finish_prepare(qx,qzz,u2rx,prod);'
assert tree.count(call) == 1
tree = tree.replace(call, 'qsb_xyzz_finish_prepare(qx,qzz,qzzz,u2rx,prod);')
old_c = '    if(usable){ _ModSqr(qx,qx); _ModMult(qx,qzz); }   /* qx -> C = ZZ*d^2 */\n'
assert tree.count(old_c) == 1
tree = tree.replace(old_c, '')
old_call = 'qsb_xyzz_finish_precomputed(qx,qy,Wsave,qzzz,prod,u2rx,u2ry,q1x,q2x)'
assert tree.count(old_call) == 1
tree = tree.replace(old_call, 'qsb_xyzz_finish_precomputed(qzz,qy,Wsave,qzzz,prod,u2rx,u2ry,q1x,q2x)')
tree = tree.replace('/* Both recovery flags from one shared-denominator inverse, in XYZZ:\n     * W = ZZ^2*d with d = xR*ZZ - X; the block inverts W. */',
                    '/* Both recovery flags use W=ZZZ*d, d=xR*ZZ-X. The block inverse and\n     * zero-denominator participation contract are unchanged. */')

host = r'''// One legitimate per-problem field square and a multiplication by3.
// Its CPU cost and constant upload remain inside process startup.
static void qsb_prepare_cubic_constant(const uint8_t x_le[32]){
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_lebin2bn(x_le,32,nullptr),*value=BN_new(),*prime=nullptr;
    compact_require(ctx&&x&&value,"cubic constant allocation");
    compact_require(BN_hex2bn(&prime,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F")!=0,
                    "cubic constant field modulus");
    compact_require(BN_mod_sqr(value,x,prime,ctx),"cubic constant square");
    compact_require(BN_mul_word(value,3),"cubic constant triple");
    compact_require(BN_nnmod(value,value,prime,ctx),"cubic constant reduction");
    uint8_t le[32];uint64_t limbs[4];
    compact_require(BN_bn2lebinpad(value,le,32)==32,"cubic constant serialization");
    memcpy(limbs,le,32);
    BN_free(x);BN_free(value);BN_free(prime);BN_CTX_free(ctx);
    wide_cuda_require(cudaMemcpyToSymbol(QSB_U2R_C3X2,limbs,sizeof(limbs)),"upload cubic recovery constant");
}

'''
marker = '/* Digest params loader */'
assert tree.count(marker) == 1
tree = tree.replace(marker, host + marker)
marker = '    /* Compute neg_2u2R */'
assert tree.count(marker) == 1
tree = tree.replace(marker, '    qsb_prepare_cubic_constant(dp.u2r_x);\n\n' + marker)
files[path] = tree
for name, data in files.items():
    p = DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
identity = source_identity(DEST)
changed = [name for name in base_id['source_sha256']
           if base_id['source_sha256'][name] != identity['source_sha256'][name]]
assert changed == [path], changed
receipt = {
    'status': 'PREPARED_FOR_VALIDATION', 'base_source': base_id, 'candidate_source': identity,
    'changed_source_files': changed,
    'changes': ['Recovery denominator B*d replaces A^2*d; inverse tree and zero-lane contract preserved.',
                'A is carried across the inverse instead of C=A*d^2.',
                'Cubic curve identity replaces two slope squares and two pre-inverse squares with one alpha square.',
                'Per-problem3*xR^2 is computed with checked OpenSSL operations and uploaded once as32constantbytes.'],
    'per_candidate_recovery_before': {'M':10,'S':4},
    'per_candidate_recovery_after': {'M':10,'S':1},
    'whole_point_chain_plus_recovery_before': {'M':98,'S':30},
    'whole_point_chain_plus_recovery_after': {'M':98,'S':27},
    'count_limits': 'Counts exclude unchanged inverse-tree arithmetic, SHA and additions. One host field square/triple/reduction and32-byte constant upload per runtime problem are additional.',
    'unchanged': ['Guarded complete small48 point chain','GPUMath and square carry corrections','per-CTA inverse',
                  'table builder and policy','candidate selection/hashing','hit flags and host search loop'],
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'gpu_executed':False,'production_changed':False,
}
(HERE/'prepared-source.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(identity['source_fingerprint'])
