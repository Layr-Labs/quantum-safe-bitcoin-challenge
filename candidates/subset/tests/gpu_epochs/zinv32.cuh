/* zinv32: 32-bit-word delayed binary-GCD field inverse for secp256k1 p.
 * Same algorithm family as VanitySearch's _ModInv (Jean Luc Pons, GPL-3.0; see GPUMath.h):
 * delayed right-shift divsteps on low/high approximations, r/s tracked with a Montgomery-style
 * m*p correction. Differences: 30-bit batches on 32-bit registers (int32 matrix), 9x32-bit
 * signed limbs, sparse m*p, sentinel-terminated decision loop. Output: canonical inverse in
 * [0,p), or 0 when gcd != 1 (input 0), bit-identical to _ModInv on canonical inputs.
 *
 * The cooperative form below (lanes owning U, V, R and S of one root inverse and exchanging
 * only the matrix rows, the sign flag, the zero flag and the partner vector) follows the
 * structure of @AbdelStark's warp-cooperative root inverse (PR 189 / submission db248c65,
 * hm41/hm43 ablations); the algorithm, the 32-bit decision core and all arithmetic here are
 * ours. Measured on an RTX 4090: one root inverse costs 39,851 cycles with the serial lane-0
 * _ModInv and 28,155 with this form (-29.4%); 85-90% of what remains is the ~190-iteration
 * divstep decision chain, which is serial in any design. */
/* ZLAB_BY (default 1) replaces that decision chain with table-driven
 * Bernstein-Yang divsteps: 7 four-step constant-memory lookups plus 2
 * branchless single steps per 30-bit batch, no ctz, no brev/flo, no branch and
 * no head alignment, at the cost of ~44% more batches (18.10 vs 12.54 mean).
 * ZLAB_BY=0 selects the Stein path below unchanged and compiles byte-identical
 * to the base.  Both produce the same canonical inverse: 314,081 cross-checked
 * inversions agree bit for bit and with OpenSSL BN_mod_inverse (zlab_by/). */
#pragma once
#ifdef __CUDA_ARCH__
#define ZI_DEV __device__ __forceinline__
ZI_DEV uint32_t zi_ctz32(uint32_t x){
    uint32_t n;
    asm("{\n\t .reg .u32 tmp;\n\t brev.b32 tmp, %1;\n\t clz.b32 %0, tmp;\n\t}" : "=r"(n) : "r"(x));
    return n;
}
ZI_DEV uint32_t zi_clz32(uint32_t x){
    uint32_t n;
    asm("{\n\t clz.b32 %0, %1;\n\t}" : "=r"(n) : "r"(x));
    return n;
}
#else
#define ZI_DEV static inline
static inline uint32_t zi_ctz32(uint32_t x){return (uint32_t)__builtin_ctz(x);}
static inline uint32_t zi_clz32(uint32_t x){return x?(uint32_t)__builtin_clz(x):32u;}
#endif
#define ZI_B 30
#define ZI_MM32 0xD2253531u            /* -p^-1 mod 2^32 */
#define ZI_MASK30 0x3FFFFFFFu

/* ZLAB_BY (kill switch): 1 = table-driven Bernstein-Yang divsteps, 0 = the
 * Stein/magnitude-comparison decision loop below (byte-identical to the base).
 *
 * A Bernstein-Yang divstep
 *     delta>0 and g odd : (delta,f,g) <- (1-delta, g, (g-f)/2)
 *     delta<=0, g odd   : (delta,f,g) <- (1+delta, f, (g+f)/2)
 *     g even            : (delta,f,g) <- (1+delta, f, g/2)
 * depends only on (delta, f mod 2^k, g mod 2^k), so k=4 steps collapse into one
 * lookup: f is always odd (8 values of f mod 16), 16 values of g mod 16 and
 * delta clamped to [-4,4] (9 values) = 1152 entries, 4608 B, one 32-bit word
 * each.  The clamp is exact: over 4 steps a delta above 4 stays positive until
 * the first swap and cannot return above 0 afterwards, and a delta below -4
 * cannot reach 0, so the clamped run makes the same decisions; the affine delta
 * recurrence then gives delta' = (-1)^swaps * delta + C with C in [-2,4] a
 * function of the clamped state alone (proved exhaustively in zlab_by/).
 *
 * Entry layout: a:6 b:6 c:6 d:6 (signed, in [-8,16]) | C:4 (signed) | bit31 =
 * parity of the swaps.  f' = (a*f+b*g)>>4, g' = (c*f+d*g)>>4 exactly.
 * The table is problem-independent, so it is a __constant__ initialiser: no
 * host upload, no shared-memory budget (the kernel is at 32,768 B shared with
 * exactly 2 resident blocks) and lanes 0 and 1 always present the same address,
 * so the LDC is a broadcast out of the constant cache. */
#ifndef ZLAB_BY
#define ZLAB_BY 1
#endif
#if ZLAB_BY
#ifdef __CUDA_ARCH__
#define ZI_CONST __constant__
#else
#define ZI_CONST static const
#endif
ZI_CONST uint32_t ZI_BY_LUT[1152]={
    0x04040010u,0x0404F010u,0x0404E010u,0x0404D010u,0x0404C010u,0x0404B010u,0x0404A010u,0x04049010u,
    0x04048010u,0x04047010u,0x04046010u,0x04045010u,0x04044010u,0x04043010u,0x04042010u,0x04041010u,
    0x04040010u,0x04045010u,0x0404A010u,0x0404F010u,0x04044010u,0x04049010u,0x0404E010u,0x04043010u,
    0x04048010u,0x0404D010u,0x04042010u,0x04047010u,0x0404C010u,0x04041010u,0x04046010u,0x0404B010u,
    0x04040010u,0x04043010u,0x04046010u,0x04049010u,0x0404C010u,0x0404F010u,0x04042010u,0x04045010u,
    0x04048010u,0x0404B010u,0x0404E010u,0x04041010u,0x04044010u,0x04047010u,0x0404A010u,0x0404D010u,
    0x04040010u,0x04049010u,0x04042010u,0x0404B010u,0x04044010u,0x0404D010u,0x04046010u,0x0404F010u,
    0x04048010u,0x04041010u,0x0404A010u,0x04043010u,0x0404C010u,0x04045010u,0x0404E010u,0x04047010u,
    0x04040010u,0x04047010u,0x0404E010u,0x04045010u,0x0404C010u,0x04043010u,0x0404A010u,0x04041010u,
    0x04048010u,0x0404F010u,0x04046010u,0x0404D010u,0x04044010u,0x0404B010u,0x04042010u,0x04049010u,
    0x04040010u,0x0404D010u,0x0404A010u,0x04047010u,0x04044010u,0x04041010u,0x0404E010u,0x0404B010u,
    0x04048010u,0x04045010u,0x04042010u,0x0404F010u,0x0404C010u,0x04049010u,0x04046010u,0x04043010u,
    0x04040010u,0x0404B010u,0x04046010u,0x04041010u,0x0404C010u,0x04047010u,0x04042010u,0x0404D010u,
    0x04048010u,0x04043010u,0x0404E010u,0x04049010u,0x04044010u,0x0404F010u,0x0404A010u,0x04045010u,
    0x04040010u,0x04041010u,0x04042010u,0x04043010u,0x04044010u,0x04045010u,0x04046010u,0x04047010u,
    0x04048010u,0x04049010u,0x0404A010u,0x0404B010u,0x0404C010u,0x0404D010u,0x0404E010u,0x0404F010u,
    0x04040010u,0x0404F010u,0x0404E010u,0x0404D010u,0x0404C010u,0x0404B010u,0x0404A010u,0x04049010u,
    0x04048010u,0x04047010u,0x04046010u,0x04045010u,0x04044010u,0x04043010u,0x04042010u,0x04041010u,
    0x04040010u,0x04045010u,0x0404A010u,0x0404F010u,0x04044010u,0x04049010u,0x0404E010u,0x04043010u,
    0x04048010u,0x0404D010u,0x04042010u,0x04047010u,0x0404C010u,0x04041010u,0x04046010u,0x0404B010u,
    0x04040010u,0x04043010u,0x04046010u,0x04049010u,0x0404C010u,0x0404F010u,0x04042010u,0x04045010u,
    0x04048010u,0x0404B010u,0x0404E010u,0x04041010u,0x04044010u,0x04047010u,0x0404A010u,0x0404D010u,
    0x04040010u,0x04049010u,0x04042010u,0x0404B010u,0x04044010u,0x0404D010u,0x04046010u,0x0404F010u,
    0x04048010u,0x04041010u,0x0404A010u,0x04043010u,0x0404C010u,0x04045010u,0x0404E010u,0x04047010u,
    0x04040010u,0x04047010u,0x0404E010u,0x04045010u,0x0404C010u,0x04043010u,0x0404A010u,0x04041010u,
    0x04048010u,0x0404F010u,0x04046010u,0x0404D010u,0x04044010u,0x0404B010u,0x04042010u,0x04049010u,
    0x04040010u,0x0404D010u,0x0404A010u,0x04047010u,0x04044010u,0x04041010u,0x0404E010u,0x0404B010u,
    0x04048010u,0x04045010u,0x04042010u,0x0404F010u,0x0404C010u,0x04049010u,0x04046010u,0x04043010u,
    0x04040010u,0x0404B010u,0x04046010u,0x04041010u,0x0404C010u,0x04047010u,0x04042010u,0x0404D010u,
    0x04048010u,0x04043010u,0x0404E010u,0x04049010u,0x04044010u,0x0404F010u,0x0404A010u,0x04045010u,
    0x04040010u,0x04041010u,0x04042010u,0x04043010u,0x04044010u,0x04045010u,0x04046010u,0x04047010u,
    0x04048010u,0x04049010u,0x0404A010u,0x0404B010u,0x0404C010u,0x0404D010u,0x0404E010u,0x0404F010u,
    0x04040010u,0x8E07F08Eu,0x8E07E08Cu,0x8E07D08Au,0x8E07C088u,0x8E07B086u,0x8E07A084u,0x8E079082u,
    0x8E078080u,0x04047010u,0x04046010u,0x04045010u,0x04044010u,0x04043010u,0x04042010u,0x04041010u,
    0x04040010u,0x04045010u,0x8E07A084u,0x8E07F08Eu,0x04044010u,0x8E079082u,0x8E07E08Cu,0x04043010u,
    0x8E078080u,0x8E07D08Au,0x04042010u,0x04047010u,0x8E07C088u,0x04041010u,0x04046010u,0x8E07B086u,
    0x04040010u,0x04043010u,0x04046010u,0x8E079082u,0x8E07C088u,0x8E07F08Eu,0x04042010u,0x04045010u,
    0x8E078080u,0x8E07B086u,0x8E07E08Cu,0x04041010u,0x04044010u,0x04047010u,0x8E07A084u,0x8E07D08Au,
    0x04040010u,0x8E079082u,0x04042010u,0x8E07B086u,0x04044010u,0x8E07D08Au,0x04046010u,0x8E07F08Eu,
    0x8E078080u,0x04041010u,0x8E07A084u,0x04043010u,0x8E07C088u,0x04045010u,0x8E07E08Cu,0x04047010u,
    0x04040010u,0x04047010u,0x8E07E08Cu,0x04045010u,0x8E07C088u,0x04043010u,0x8E07A084u,0x04041010u,
    0x8E078080u,0x8E07F08Eu,0x04046010u,0x8E07D08Au,0x04044010u,0x8E07B086u,0x04042010u,0x8E079082u,
    0x04040010u,0x8E07D08Au,0x8E07A084u,0x04047010u,0x04044010u,0x04041010u,0x8E07E08Cu,0x8E07B086u,
    0x8E078080u,0x04045010u,0x04042010u,0x8E07F08Eu,0x8E07C088u,0x8E079082u,0x04046010u,0x04043010u,
    0x04040010u,0x8E07B086u,0x04046010u,0x04041010u,0x8E07C088u,0x04047010u,0x04042010u,0x8E07D08Au,
    0x8E078080u,0x04043010u,0x8E07E08Cu,0x8E079082u,0x04044010u,0x8E07F08Eu,0x8E07A084u,0x04045010u,
    0x04040010u,0x04041010u,0x04042010u,0x04043010u,0x04044010u,0x04045010u,0x04046010u,0x04047010u,
    0x8E078080u,0x8E079082u,0x8E07A084u,0x8E07B086u,0x8E07C088u,0x8E07D08Au,0x8E07E08Cu,0x8E07F08Eu,
    0x04040010u,0x8007F10Cu,0x8007E108u,0x8007D104u,0x8007C100u,0x8E07B086u,0x8E07A084u,0x8E079082u,
    0x8E078080u,0x800C510Cu,0x800C2108u,0x800FF104u,0x800FC100u,0x04043010u,0x04042010u,0x04041010u,
    0x04040010u,0x800FF104u,0x8E07A084u,0x8007F10Cu,0x800FC100u,0x8E079082u,0x8007E108u,0x04043010u,
    0x8E078080u,0x8007D104u,0x04042010u,0x800C510Cu,0x8007C100u,0x04041010u,0x800C2108u,0x8E07B086u,
    0x04040010u,0x04043010u,0x800C2108u,0x8E079082u,0x8007C100u,0x8007F10Cu,0x04042010u,0x800FF104u,
    0x8E078080u,0x8E07B086u,0x8007E108u,0x04041010u,0x800FC100u,0x800C510Cu,0x8E07A084u,0x8007D104u,
    0x04040010u,0x8E079082u,0x04042010u,0x8E07B086u,0x800FC100u,0x8007D104u,0x800C2108u,0x8007F10Cu,
    0x8E078080u,0x04041010u,0x8E07A084u,0x04043010u,0x8007C100u,0x800FF104u,0x8007E108u,0x800C510Cu,
    0x04040010u,0x800C510Cu,0x8007E108u,0x800FF104u,0x8007C100u,0x04043010u,0x8E07A084u,0x04041010u,
    0x8E078080u,0x8007F10Cu,0x800C2108u,0x8007D104u,0x800FC100u,0x8E07B086u,0x04042010u,0x8E079082u,
    0x04040010u,0x8007D104u,0x8E07A084u,0x800C510Cu,0x800FC100u,0x04041010u,0x8007E108u,0x8E07B086u,
    0x8E078080u,0x800FF104u,0x04042010u,0x8007F10Cu,0x8007C100u,0x8E079082u,0x800C2108u,0x04043010u,
    0x04040010u,0x8E07B086u,0x800C2108u,0x04041010u,0x8007C100u,0x800C510Cu,0x04042010u,0x8007D104u,
    0x8E078080u,0x04043010u,0x8007E108u,0x8E079082u,0x800FC100u,0x8007F10Cu,0x8E07A084u,0x800FF104u,
    0x04040010u,0x04041010u,0x04042010u,0x04043010u,0x800FC100u,0x800FF104u,0x800C2108u,0x800C510Cu,
    0x8E078080u,0x8E079082u,0x8E07A084u,0x8E07B086u,0x8007C100u,0x8007D104u,0x8007E108u,0x8007F10Cu,
    0x04040010u,0x8207F208u,0x8207E200u,0x8007D104u,0x8007C100u,0x820C1208u,0x820FE200u,0x8E079082u,
    0x8E078080u,0x00F7B0BEu,0x00F7E0BCu,0x800FF104u,0x800FC100u,0x00FFD182u,0x00FFE1BCu,0x04041010u,
    0x04040010u,0x800FF104u,0x820FE200u,0x8207F208u,0x800FC100u,0x8E079082u,0x8207E200u,0x00FFD182u,
    0x8E078080u,0x8007D104u,0x00FFE1BCu,0x00F7B0BEu,0x8007C100u,0x04041010u,0x00F7E0BCu,0x820C1208u,
    0x04040010u,0x00FFD182u,0x00F7E0BCu,0x8E079082u,0x8007C100u,0x8207F208u,0x00FFE1BCu,0x800FF104u,
    0x8E078080u,0x820C1208u,0x8207E200u,0x04041010u,0x800FC100u,0x00F7B0BEu,0x820FE200u,0x8007D104u,
    0x04040010u,0x8E079082u,0x00FFE1BCu,0x820C1208u,0x800FC100u,0x8007D104u,0x00F7E0BCu,0x8207F208u,
    0x8E078080u,0x04041010u,0x820FE200u,0x00FFD182u,0x8007C100u,0x800FF104u,0x8207E200u,0x00F7B0BEu,
    0x04040010u,0x00F7B0BEu,0x8207E200u,0x800FF104u,0x8007C100u,0x00FFD182u,0x820FE200u,0x04041010u,
    0x8E078080u,0x8207F208u,0x00F7E0BCu,0x8007D104u,0x800FC100u,0x820C1208u,0x00FFE1BCu,0x8E079082u,
    0x04040010u,0x8007D104u,0x820FE200u,0x00F7B0BEu,0x800FC100u,0x04041010u,0x8207E200u,0x820C1208u,
    0x8E078080u,0x800FF104u,0x00FFE1BCu,0x8207F208u,0x8007C100u,0x8E079082u,0x00F7E0BCu,0x00FFD182u,
    0x04040010u,0x820C1208u,0x00F7E0BCu,0x04041010u,0x8007C100u,0x00F7B0BEu,0x00FFE1BCu,0x8007D104u,
    0x8E078080u,0x00FFD182u,0x8207E200u,0x8E079082u,0x800FC100u,0x8207F208u,0x820FE200u,0x800FF104u,
    0x04040010u,0x04041010u,0x00FFE1BCu,0x00FFD182u,0x800FC100u,0x800FF104u,0x00F7E0BCu,0x00F7B0BEu,
    0x8E078080u,0x8E079082u,0x820FE200u,0x820C1208u,0x8007C100u,0x8007D104u,0x8207E200u,0x8207F208u,
    0x04040010u,0x8407F400u,0x8207E200u,0x0EEFF1BEu,0x8007C100u,0x00F7F13Cu,0x820FE200u,0x0017D33Cu,
    0x8E078080u,0x0EE7F0BEu,0x8217E200u,0x840FF400u,0x800FC100u,0x00FFD13Cu,0x821FE200u,0x00FFF33Cu,
    0x04040010u,0x840FF400u,0x820FE200u,0x8407F400u,0x800FC100u,0x0017D33Cu,0x8207E200u,0x00FFD13Cu,
    0x8E078080u,0x0EEFF1BEu,0x821FE200u,0x0EE7F0BEu,0x8007C100u,0x00FFF33Cu,0x8217E200u,0x00F7F13Cu,
    0x04040010u,0x00FFD13Cu,0x8217E200u,0x0017D33Cu,0x8007C100u,0x8407F400u,0x821FE200u,0x840FF400u,
    0x8E078080u,0x00F7F13Cu,0x8207E200u,0x00FFF33Cu,0x800FC100u,0x0EE7F0BEu,0x820FE200u,0x0EEFF1BEu,
    0x04040010u,0x0017D33Cu,0x821FE200u,0x00F7F13Cu,0x800FC100u,0x0EEFF1BEu,0x8217E200u,0x8407F400u,
    0x8E078080u,0x00FFF33Cu,0x820FE200u,0x00FFD13Cu,0x8007C100u,0x840FF400u,0x8207E200u,0x0EE7F0BEu,
    0x04040010u,0x0EE7F0BEu,0x8207E200u,0x840FF400u,0x8007C100u,0x00FFD13Cu,0x820FE200u,0x00FFF33Cu,
    0x8E078080u,0x8407F400u,0x8217E200u,0x0EEFF1BEu,0x800FC100u,0x00F7F13Cu,0x821FE200u,0x0017D33Cu,
    0x04040010u,0x0EEFF1BEu,0x820FE200u,0x0EE7F0BEu,0x800FC100u,0x00FFF33Cu,0x8207E200u,0x00F7F13Cu,
    0x8E078080u,0x840FF400u,0x821FE200u,0x8407F400u,0x8007C100u,0x0017D33Cu,0x8217E200u,0x00FFD13Cu,
    0x04040010u,0x00F7F13Cu,0x8217E200u,0x00FFF33Cu,0x8007C100u,0x0EE7F0BEu,0x821FE200u,0x0EEFF1BEu,
    0x8E078080u,0x00FFD13Cu,0x8207E200u,0x0017D33Cu,0x800FC100u,0x8407F400u,0x820FE200u,0x840FF400u,
    0x04040010u,0x00FFF33Cu,0x821FE200u,0x00FFD13Cu,0x800FC100u,0x840FF400u,0x8217E200u,0x0EE7F0BEu,
    0x8E078080u,0x0017D33Cu,0x820FE200u,0x00F7F13Cu,0x8007C100u,0x0EEFF1BEu,0x8207E200u,0x8407F400u,
    0x04040010u,0x8407F400u,0x8207E200u,0x0EEFF1BEu,0x8007C100u,0x0EF7F2BEu,0x820FE200u,0x841FF400u,
    0x8E078080u,0x0EE7F0BEu,0x8217E200u,0x840FF400u,0x800FC100u,0x8417F400u,0x821FE200u,0x0EFFF3BEu,
    0x04040010u,0x840FF400u,0x820FE200u,0x8407F400u,0x800FC100u,0x841FF400u,0x8207E200u,0x8417F400u,
    0x8E078080u,0x0EEFF1BEu,0x821FE200u,0x0EE7F0BEu,0x8007C100u,0x0EFFF3BEu,0x8217E200u,0x0EF7F2BEu,
    0x04040010u,0x8417F400u,0x8217E200u,0x841FF400u,0x8007C100u,0x8407F400u,0x821FE200u,0x840FF400u,
    0x8E078080u,0x0EF7F2BEu,0x8207E200u,0x0EFFF3BEu,0x800FC100u,0x0EE7F0BEu,0x820FE200u,0x0EEFF1BEu,
    0x04040010u,0x841FF400u,0x821FE200u,0x0EF7F2BEu,0x800FC100u,0x0EEFF1BEu,0x8217E200u,0x8407F400u,
    0x8E078080u,0x0EFFF3BEu,0x820FE200u,0x8417F400u,0x8007C100u,0x840FF400u,0x8207E200u,0x0EE7F0BEu,
    0x04040010u,0x0EE7F0BEu,0x8207E200u,0x840FF400u,0x8007C100u,0x8417F400u,0x820FE200u,0x0EFFF3BEu,
    0x8E078080u,0x8407F400u,0x8217E200u,0x0EEFF1BEu,0x800FC100u,0x0EF7F2BEu,0x821FE200u,0x841FF400u,
    0x04040010u,0x0EEFF1BEu,0x820FE200u,0x0EE7F0BEu,0x800FC100u,0x0EFFF3BEu,0x8207E200u,0x0EF7F2BEu,
    0x8E078080u,0x840FF400u,0x821FE200u,0x8407F400u,0x8007C100u,0x841FF400u,0x8217E200u,0x8417F400u,
    0x04040010u,0x0EF7F2BEu,0x8217E200u,0x0EFFF3BEu,0x8007C100u,0x0EE7F0BEu,0x821FE200u,0x0EEFF1BEu,
    0x8E078080u,0x8417F400u,0x8207E200u,0x841FF400u,0x800FC100u,0x8407F400u,0x820FE200u,0x840FF400u,
    0x04040010u,0x0EFFF3BEu,0x821FE200u,0x8417F400u,0x800FC100u,0x840FF400u,0x8217E200u,0x0EE7F0BEu,
    0x8E078080u,0x841FF400u,0x820FE200u,0x0EF7F2BEu,0x8007C100u,0x0EEFF1BEu,0x8207E200u,0x8407F400u,
    0x04040010u,0x8407F400u,0x8207E200u,0x842FF400u,0x8007C100u,0x8437F400u,0x820FE200u,0x841FF400u,
    0x8E078080u,0x8427F400u,0x8217E200u,0x840FF400u,0x800FC100u,0x8417F400u,0x821FE200u,0x843FF400u,
    0x04040010u,0x840FF400u,0x820FE200u,0x8407F400u,0x800FC100u,0x841FF400u,0x8207E200u,0x8417F400u,
    0x8E078080u,0x842FF400u,0x821FE200u,0x8427F400u,0x8007C100u,0x843FF400u,0x8217E200u,0x8437F400u,
    0x04040010u,0x8417F400u,0x8217E200u,0x841FF400u,0x8007C100u,0x8407F400u,0x821FE200u,0x840FF400u,
    0x8E078080u,0x8437F400u,0x8207E200u,0x843FF400u,0x800FC100u,0x8427F400u,0x820FE200u,0x842FF400u,
    0x04040010u,0x841FF400u,0x821FE200u,0x8437F400u,0x800FC100u,0x842FF400u,0x8217E200u,0x8407F400u,
    0x8E078080u,0x843FF400u,0x820FE200u,0x8417F400u,0x8007C100u,0x840FF400u,0x8207E200u,0x8427F400u,
    0x04040010u,0x8427F400u,0x8207E200u,0x840FF400u,0x8007C100u,0x8417F400u,0x820FE200u,0x843FF400u,
    0x8E078080u,0x8407F400u,0x8217E200u,0x842FF400u,0x800FC100u,0x8437F400u,0x821FE200u,0x841FF400u,
    0x04040010u,0x842FF400u,0x820FE200u,0x8427F400u,0x800FC100u,0x843FF400u,0x8207E200u,0x8437F400u,
    0x8E078080u,0x840FF400u,0x821FE200u,0x8407F400u,0x8007C100u,0x841FF400u,0x8217E200u,0x8417F400u,
    0x04040010u,0x8437F400u,0x8217E200u,0x843FF400u,0x8007C100u,0x8427F400u,0x821FE200u,0x842FF400u,
    0x8E078080u,0x8417F400u,0x8207E200u,0x841FF400u,0x800FC100u,0x8407F400u,0x820FE200u,0x840FF400u,
    0x04040010u,0x843FF400u,0x821FE200u,0x8417F400u,0x800FC100u,0x840FF400u,0x8217E200u,0x8427F400u,
    0x8E078080u,0x841FF400u,0x820FE200u,0x8437F400u,0x8007C100u,0x842FF400u,0x8207E200u,0x8407F400u,
    0x04040010u,0x8407F400u,0x8207E200u,0x842FF400u,0x8007C100u,0x8437F400u,0x820FE200u,0x841FF400u,
    0x8E078080u,0x8427F400u,0x8217E200u,0x840FF400u,0x800FC100u,0x8417F400u,0x821FE200u,0x843FF400u,
    0x04040010u,0x840FF400u,0x820FE200u,0x8407F400u,0x800FC100u,0x841FF400u,0x8207E200u,0x8417F400u,
    0x8E078080u,0x842FF400u,0x821FE200u,0x8427F400u,0x8007C100u,0x843FF400u,0x8217E200u,0x8437F400u,
    0x04040010u,0x8417F400u,0x8217E200u,0x841FF400u,0x8007C100u,0x8407F400u,0x821FE200u,0x840FF400u,
    0x8E078080u,0x8437F400u,0x8207E200u,0x843FF400u,0x800FC100u,0x8427F400u,0x820FE200u,0x842FF400u,
    0x04040010u,0x841FF400u,0x821FE200u,0x8437F400u,0x800FC100u,0x842FF400u,0x8217E200u,0x8407F400u,
    0x8E078080u,0x843FF400u,0x820FE200u,0x8417F400u,0x8007C100u,0x840FF400u,0x8207E200u,0x8427F400u,
    0x04040010u,0x8427F400u,0x8207E200u,0x840FF400u,0x8007C100u,0x8417F400u,0x820FE200u,0x843FF400u,
    0x8E078080u,0x8407F400u,0x8217E200u,0x842FF400u,0x800FC100u,0x8437F400u,0x821FE200u,0x841FF400u,
    0x04040010u,0x842FF400u,0x820FE200u,0x8427F400u,0x800FC100u,0x843FF400u,0x8207E200u,0x8437F400u,
    0x8E078080u,0x840FF400u,0x821FE200u,0x8407F400u,0x8007C100u,0x841FF400u,0x8217E200u,0x8417F400u,
    0x04040010u,0x8437F400u,0x8217E200u,0x843FF400u,0x8007C100u,0x8427F400u,0x821FE200u,0x842FF400u,
    0x8E078080u,0x8417F400u,0x8207E200u,0x841FF400u,0x800FC100u,0x8407F400u,0x820FE200u,0x840FF400u,
    0x04040010u,0x843FF400u,0x821FE200u,0x8417F400u,0x800FC100u,0x840FF400u,0x8217E200u,0x8427F400u,
    0x8E078080u,0x841FF400u,0x820FE200u,0x8437F400u,0x8007C100u,0x842FF400u,0x8207E200u,0x8407F400u
};
/* 30 Bernstein-Yang divsteps on the low words of f and g: 7 four-step lookups
 * plus 2 branchless single steps.  No ctz, no head alignment, no branch on the
 * decision path.  Returns the new delta and the matrix rows (a,b) for f and
 * (c,d) for g, each row l1-norm <= 2^30 (so zi_row_ip's int32 x uint32 -> int64
 * products stay below 2^62, exactly as for the Stein rows). */
ZI_DEV int32_t zi_divstep30_by(int32_t delta,uint32_t f,uint32_t g,
                               int32_t *ra,int32_t *rb,int32_t *rc,int32_t *rd){
    int32_t u=1,v=0,q=0,r=1;
    #pragma unroll
    for(int k=0;k<7;k++){
        const int32_t dc=delta<-4?-4:(delta>4?4:delta);
        const uint32_t e=ZI_BY_LUT[(uint32_t)((dc+4)<<7)|((f&14u)<<3)|(g&15u)];
        const int32_t a=(int32_t)(e<<26)>>26,b=(int32_t)(e<<20)>>26;
        const int32_t c=(int32_t)(e<<14)>>26,d=(int32_t)(e<<8)>>26;
        const uint32_t nf=((uint32_t)a*f+(uint32_t)b*g)>>4;
        g=((uint32_t)c*f+(uint32_t)d*g)>>4; f=nf;
        const int32_t nu=a*u+b*q,nv=a*v+b*r;
        q=c*u+d*q; r=c*v+d*r; u=nu; v=nv;
        const int32_t sm=(int32_t)e>>31;
        delta=((delta^sm)-sm)+((int32_t)(e<<4)>>28);
    }
    #pragma unroll
    for(int k=0;k<2;k++){
        const int32_t mg=-(int32_t)(g&1u);
        const int32_t sw=mg&-(int32_t)(delta>0);
        const uint32_t nf=f^((f^g)&(uint32_t)sw);
        int32_t t=(int32_t)f&mg; t=(t^sw)-sw;
        g=(g+(uint32_t)t)>>1; f=nf;
        int32_t tu=u&mg; tu=(tu^sw)-sw;
        int32_t tv=v&mg; tv=(tv^sw)-sw;
        const int32_t nu=((sw&(q^u))^u)*2,nv=((sw&(r^v))^v)*2;
        q+=tu; r+=tv; u=nu; v=nv;
        delta=((delta^sw)-sw)+1;
    }
    *ra=u;*rb=v;*rc=q;*rd=r;
    return delta;
}
#else

/* Decision loop: 30 delayed divsteps on (u0,v0) low words and (uh,vh) aligned heads.
 * Returns matrix rows (a,b) for u and (c,d) for v, each row l1-norm <= 2^30. */
ZI_DEV void zi_divstep30(uint32_t u0,uint32_t v0,uint32_t uh,uint32_t vh,
                         int32_t *ra,int32_t *rb,int32_t *rc,int32_t *rd){
    uint32_t a=1,b=0,c=0,d=1,S=1u<<ZI_B;
    while(true){
        uint32_t z=zi_ctz32(v0|S);
        v0>>=z; vh>>=z; a<<=z; b<<=z; S>>=z;
        if(S==1u)break;
        if(vh<uh){
            uint32_t t;
            t=uh;uh=vh;vh=t; t=u0;u0=v0;v0=t; t=a;a=c;c=t; t=b;b=d;d=t;
        }
        vh-=uh; v0-=u0; d-=b; c-=a;
    }
    *ra=(int32_t)a;*rb=(int32_t)b;*rc=(int32_t)c;*rd=(int32_t)d;
}
#endif  /* ZLAB_BY */

/* ======================= 4-lane cooperative form =======================
 * Lanes 0..3 of one warp run the SAME instruction stream except the decision loop
 * (lanes 0,1 only; they hold identical u,v so they take identical branches).
 * lane 0 owns u, lane 1 v, lane 2 r, lane 3 s; P = own vector, Q = pair partner's.
 * Row rule: lane even  new P = a*P + b*Q ; lane odd new P = d*P + c*Q  (same as c*u+d*v).
 * Collectives per batch: 2 coefficients + 1 sign + 1 zero flag + 9 partner limbs. */
#ifdef __CUDA_ARCH__
ZI_DEV uint32_t zi_x(uint32_t v,int src){return (uint32_t)__shfl_sync(0xFu,(unsigned int)v,src);}
#else
uint32_t zi_x(uint32_t v,int src);
#endif
/* p limbs, little-endian 32-bit (limb 8 = 0). Kept as an initialised local array in every
 * user so it folds to immediates in device code instead of a constant-bank load. */
#define ZI_PL_INIT {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0u}

/* In place: X = (a*X + b*Y [+ m*p]) >> 30. */
ZI_DEV void zi_row_ip(uint32_t *X,const uint32_t *Y,int32_t a,int32_t b,uint32_t modp){
    int64_t acc=(int64_t)a*(int64_t)X[0]+(int64_t)b*(int64_t)Y[0];
    uint32_t m=((uint32_t)acc*ZI_MM32)&ZI_MASK30&(0u-modp);
    acc-=(int64_t)977*(int64_t)m;
    X[0]=(uint32_t)acc; acc>>=32;
    acc+=(int64_t)a*(int64_t)X[1]+(int64_t)b*(int64_t)Y[1]-(int64_t)m;
    X[1]=(uint32_t)acc; acc>>=32;
    for(int i=2;i<8;i++){
        acc+=(int64_t)a*(int64_t)X[i]+(int64_t)b*(int64_t)Y[i];
        X[i]=(uint32_t)acc; acc>>=32;
    }
    acc+=(int64_t)a*(int64_t)(int32_t)X[8]+(int64_t)b*(int64_t)(int32_t)Y[8]+(int64_t)m;
    X[8]=(uint32_t)acc;
    for(int i=0;i<8;i++)X[i]=(X[i]>>ZI_B)|(X[i+1]<<(32-ZI_B));
    X[8]=(uint32_t)((int32_t)X[8]>>ZI_B);
}
ZI_DEV void zi_condneg(uint32_t *X,uint32_t neg){
    const uint32_t msk=0u-neg; uint64_t c=neg;
    for(int i=0;i<9;i++){c+=(uint64_t)(X[i]^msk);X[i]=(uint32_t)c;c>>=32;}
}
/* X signed 288-bit with |X| < 2^261 and X == r (mod p) -> canonical r in X[0..7]. */
ZI_DEV void zi_canon(uint32_t *X){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    const int32_t hi=(int32_t)X[8];
    int64_t acc=(int64_t)X[0]+(int64_t)hi*977;
    X[0]=(uint32_t)acc; acc>>=32;
    acc+=(int64_t)X[1]+(int64_t)hi;
    X[1]=(uint32_t)acc; acc>>=32;
    for(int i=2;i<8;i++){acc+=(int64_t)X[i];X[i]=(uint32_t)acc;acc>>=32;}
    X[8]=(uint32_t)acc;                               /* -1, 0 or 1 */
    const uint32_t mneg=(uint32_t)((int32_t)X[8]>>31);
    uint64_t c=0;
    for(int i=0;i<9;i++){c+=(uint64_t)X[i]+(uint64_t)(ZI_PL[i]&mneg);X[i]=(uint32_t)c;c>>=32;}
    uint32_t T[9]; c=1;
    for(int i=0;i<9;i++){c+=(uint64_t)X[i]+(uint64_t)(uint32_t)~ZI_PL[i];T[i]=(uint32_t)c;c>>=32;}
    const uint32_t keep=(uint32_t)((int32_t)T[8]>>31);
    for(int i=0;i<8;i++)X[i]=(X[i]&keep)|(T[i]&~keep);
}
#if ZLAB_BY
/* All four lanes pass the same canonical root in R[0..3]; all return the canonical inverse
 * (0 for root 0: g starts at 0, one batch gives R=0, canon(0)=0 -- no gcd test needed since
 * p is prime).  Lane 0 owns f (init p), lane 1 g (init x), lane 2 R (init 0), lane 3 S (init 1);
 * the invariant f == R*x and g == S*x (mod p) is preserved because (R,S) take the same matrix
 * as (f,g) with the m*p correction supplying the division by 2^30 modulo p.  The loop exits
 * when g == 0, at which point f == +-gcd(p,x) == +-1, so R == +-1/x: hence the single final
 * conditional negate of R driven by the sign of lane 0's f.
 * Unlike the Stein path there is NO per-batch zi_condneg: Bernstein-Yang needs signed f and g
 * (negating g alone changes the trajectory), and none of what remains wants them non-negative
 * -- zi_row_ip already reads limb 8 as int32 and treats X as a 288-bit two's-complement value,
 * head alignment (the only magnitude comparison) is gone, and max(|f|,|g|) is non-increasing
 * from p, while |R|,|S| grow by at most p per batch (|a|+|b| <= 2^30, 0 <= m < 2^30), so after
 * the worst case of 25 batches (Bernstein-Yang's 741-divstep bound for 256-bit inputs, 30 per
 * batch) |R| < 26p < 2^261, inside zi_canon's stated precondition. */
ZI_DEV void zi_inverse_quad(uint64_t *R,int lane){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    uint32_t P[9],Q[9];
    const uint32_t odd=(uint32_t)(lane&1),rs=(uint32_t)((lane>>1)&1);
    #pragma unroll
    for(int i=0;i<9;i++){
        const uint32_t xl=i<8?(uint32_t)(R[i>>1]>>(32*(i&1))):0u;
        const uint32_t own=rs?(uint32_t)(i==0):xl;        /* S=1 / g=x */
        const uint32_t oth=rs?0u:ZI_PL[i];                 /* R=0 / f=p */
        P[i]=odd?own:oth; Q[i]=odd?oth:own;
    }
    int32_t delta=1;
    while(true){
        int32_t a=0,b=0,c=0,d=0;
        if(lane<2){
            /* both lanes see the same (delta,f0,g0), so the decision is a single
             * uniform instruction stream (no divergence, no cross-lane traffic). */
            const uint32_t f0=odd?Q[0]:P[0],g0=odd?P[0]:Q[0];
            delta=zi_divstep30_by(delta,f0,g0,&a,&b,&c,&d);
        }
        int32_t ka=odd?d:a,kb=odd?c:b;
        ka=(int32_t)zi_x((uint32_t)ka,lane&1);
        kb=(int32_t)zi_x((uint32_t)kb,lane&1);
        zi_row_ip(P,Q,ka,kb,rs);
        uint32_t nz=0;
        for(int i=0;i<9;i++)nz|=P[i];
        nz=zi_x(nz,1);
        if(nz==0)break;
        for(int i=0;i<9;i++)Q[i]=zi_x(P[i],lane^1);
    }
    uint32_t fneg=(uint32_t)((int32_t)P[8]<0);
    fneg=zi_x(fneg,0);
    zi_condneg(P,fneg);
    zi_canon(P);
    for(int i=0;i<8;i++)P[i]=zi_x(P[i],2);
    for(int i=0;i<4;i++)R[i]=(uint64_t)P[2*i]|((uint64_t)P[2*i+1]<<32);
    R[4]=0;
}
#else
/* All four lanes pass the same canonical root in R[0..3]; all return the canonical inverse
 * (0 for root 0: v starts at 0, one batch gives r=0, canon(0)=0 -- no gcd test needed since p is prime). */
ZI_DEV void zi_inverse_quad(uint64_t *R,int lane){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    uint32_t P[9],Q[9];
    const uint32_t odd=(uint32_t)(lane&1),rs=(uint32_t)((lane>>1)&1);
    #pragma unroll
    for(int i=0;i<9;i++){
        const uint32_t xl=i<8?(uint32_t)(R[i>>1]>>(32*(i&1))):0u;
        const uint32_t own=rs?(uint32_t)(i==0):xl;        /* s=1 / x */
        const uint32_t oth=rs?0u:ZI_PL[i];                 /* r=0 / p */
        P[i]=odd?own:oth; Q[i]=odd?oth:own;
    }
    int pos=7;
    while(true){
        int32_t a=0,b=0,c=0,d=0;
        if(lane<2){
            while(pos>0 && (P[pos]|Q[pos])==0)pos--;
            uint32_t ph=P[pos],qh=Q[pos];
            if(pos>0){
                const uint32_t sh=zi_clz32(ph|qh);
                if(sh){ph=(ph<<sh)|(P[pos-1]>>(32-sh));qh=(qh<<sh)|(Q[pos-1]>>(32-sh));}
            }
            /* one call, selected operands: both lanes see the same (u0,v0,uh,vh), so the
             * decision loop is a single uniform instruction stream (no divergence). */
            const uint32_t u0=odd?Q[0]:P[0],v0=odd?P[0]:Q[0],uh=odd?qh:ph,vh=odd?ph:qh;
            zi_divstep30(u0,v0,uh,vh,&a,&b,&c,&d);
        }
        int32_t ka=odd?d:a,kb=odd?c:b;
        ka=(int32_t)zi_x((uint32_t)ka,lane&1);
        kb=(int32_t)zi_x((uint32_t)kb,lane&1);
        zi_row_ip(P,Q,ka,kb,rs);
        uint32_t neg=(uint32_t)((int32_t)P[8]<0);
        neg=zi_x(neg,lane&1);
        zi_condneg(P,neg);
        uint32_t nz=0;
        for(int i=0;i<9;i++)nz|=P[i];
        nz=zi_x(nz,1);
        if(nz==0)break;
        for(int i=0;i<9;i++)Q[i]=zi_x(P[i],lane^1);
    }
    zi_canon(P);
    for(int i=0;i<8;i++)P[i]=zi_x(P[i],2);
    for(int i=0;i<4;i++)R[i]=(uint64_t)P[2*i]|((uint64_t)P[2*i+1]<<32);
    R[4]=0;
}
#endif  /* ZLAB_BY */
