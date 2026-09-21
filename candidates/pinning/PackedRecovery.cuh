#ifndef QSB_PREP_STORE_PTR
#define QSB_PREP_STORE_PTR 1 /* the prepare writes its four saved planes through one running pointer */
#endif
#ifndef QSB_PREP_NO_PUBSYNC
#define QSB_PREP_NO_PUBSYNC 1 /* the tree hand-off keeps only the publication barrier */
#endif
#ifndef QSB_REC_RAW_S
#define QSB_REC_RAW_S 1 /* the two abscissa products stay raw into the "+a" boundary */
#endif
#ifndef QSB_REC_SUM2U
#define QSB_REC_SUM2U 1 /* the finish forms the slope sum from the tbar product alone */
#endif
#if QSB_PREP_NO_PUBSYNC
#if !(QSB_CODE_QUAD && QSB_CODE_PACK16 && QSB_DIGIT_SEED_REG && QSB_DIGIT_SEED2_REG && QSB_PREP_NO_SCRATCH)
#error "the elided hand-off barrier relies on the quad arena layout with register-held seeds"
#endif
static_assert(QSB_RECOVERY_N==QSB_TREE_N,"the tree leaf slots must be the chain's lane slots");
#endif
// Let B=2^256, p=B-K, K=2^32+977. A raw exact product is in [0,B).
// If b[3]!=0 then b>=2^192>K, so -p<raw-b<p. Its canonical parity
// needs just the subtraction borrow. Small b retains normalization.
__device__ __forceinline__ void qsb_parity_boundary(uint64_t *raw,const uint64_t *b) {
    if(b[3]==0)qsb_field_normalize(raw);
}
// If a[3]!=UINT64_MAX then a<B-2^192<B-2K, hence raw+a<2p.
// One conditional subtraction in _ModAdd256 is then sufficient. The extreme
// upper fixed-a range retains the original canonical-product boundary.
__device__ __forceinline__ void qsb_add_boundary(uint64_t *raw,const uint64_t *a) {
    if(a[3]==UINT64_MAX)qsb_field_normalize(raw);
}

// Canonical inputs a,b<p. Adding odd p on borrow flips only the
// parity we need; computing all four corrected output limbs is unnecessary.
__device__ __forceinline__ uint32_t qsb_difference_parity(
    const uint64_t *a,const uint64_t *b) {
    const bool borrow=a[3]!=b[3] ? a[3]<b[3] :
        a[2]!=b[2] ? a[2]<b[2] : a[1]!=b[1] ? a[1]<b[1] : a[0]<b[0];
    return (uint32_t)((a[0]^b[0]^uint64_t(borrow))&1u);
}

#if QSB_PARITY_SUM
// P9 parity of (+-(w+b)) mod p for raw w in [0,2^256) and canonical b != 0 (b = y(u2*R): secp256k1
// has no point with y = 0). With c the carry of w+b and S its low 256 bits, w+b = y + k*p where
// y = (w+b) mod p and k in {0,1,2}: k = c unless S[1..3] are all ones (a 2^-192 event), where
// k = c + [S >= (c ? 2^256-2K : p)] and y = 0 exactly when S equals that bound. p is odd, so
// par(y) = (w0^b0^k)&1 and par(-y mod p) = y ? 1^par(y) : 0.
__device__ __forceinline__ uint32_t qsb_sum_parity(const uint64_t *w,const uint64_t *b,uint32_t neg) {
    uint64_t s0,s1,s2,s3,c;
    asm("add.cc.u64 %0,%5,%9;\n\taddc.cc.u64 %1,%6,%10;\n\taddc.cc.u64 %2,%7,%11;\n\t"
        "addc.cc.u64 %3,%8,%12;\n\taddc.u64 %4,0,0;"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(c)
        : "l"(w[0]),"l"(w[1]),"l"(w[2]),"l"(w[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    uint32_t k=(uint32_t)c,zero=0u;
    if((s1&s2&s3)==UINT64_MAX) {
        const uint64_t lim=c ? 0xFFFFFFFDFFFFF85EULL : 0xFFFFFFFEFFFFFC2FULL;
        if(s0>=lim){k+=1u;zero=(s0==lim);}
    }
    const uint32_t par=((uint32_t)(w[0]^b[0])^k)&1u;
    return zero ? 0u : (par^neg);
}
#endif

// Exact full-width residue; callers normalize before additions/parity.
__device__ __forceinline__ void qsb_packed_raw_mul(
    uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];qsb_field_mul_sc(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    Load256(out,tmp);
}

// Combine the public cofactor traversal with our existing exact/canonical
// recovery boundary and the odinfree square-free finish identity.
__device__ __forceinline__ void qsb_packed_prepare(
    uint64_t *D, const uint64_t *U, const uint64_t *Y, const uint64_t *V,
    bool usable, bool active, int n, ulonglong2 *saved, uint64_t *roots) {
#if QSB_PREP_NO_PUBSYNC
    /* The leaf publication below writes, for lane t, arena words t, 2N+t, 4N+t
     * and 6N+t (products is [4][2N] over the same storage). Under the quad
     * code layout the only live arena words at this point are the three group
     * words q*N+t, q in {0,1,2}, of each lane -- the seed and sign codes are
     * registers and the chain takes no scratch plane -- so of the four words
     * this lane stores, t and 2N+t are its own group-0 and group-2 code words,
     * which it has already consumed in its own chain, and 4N+t, 6N+t are
     * unused. No lane can therefore destroy a word another lane still owes a
     * read to, and the barrier that separated the chain from the publication
     * is redundant: the publication barrier inside qsb_cofactor_prepare still
     * orders every cross-lane read (the first of which is the partner leaf of
     * the peeled up-sweep level), and every later tree level that writes
     * another lane's former code words sits behind it. */
#else
    // All lanes finish reading their digits/anchor before tree overwrites.
    __syncthreads();
#endif
    uint64_t (*products)[2*QSB_RECOVERY_N]=(uint64_t (*)[2*QSB_RECOVERY_N])qsb_digit_arena();
    uint64_t (*excluded)[QSB_RECOVERY_N]=(uint64_t (*)[QSB_RECOVERY_N])(qsb_digit_arena()+8*QSB_TREE_N);
    qsb_cofactor_prepare<QSB_RECOVERY_N>(D,roots,products,excluded);
    if(active) {
        uint64_t hc[4],vbar[4],tbar[4];
#if QSB_PREP_ZERO_HC
        /* An unusable lane owes both saved planes zero. Clearing the common
         * factor instead of the two results does it with four selects rather
         * than eight, and the stores no longer wait on a select after the
         * products: _ModMultCore is a sum of limb products with no additive
         * term, so a zero operand yields exactly the four zero limbs the
         * explicit zeroing wrote, and the stage-2 gate on tbar is unchanged. */
        qsb_packed_raw_mul(hc,U,D);
        if(!usable)for(int k=0;k<4;k++)hc[k]=0;
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
#else
        qsb_packed_raw_mul(hc,U,D);
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
        if(!usable)for(int k=0;k<4;k++){vbar[k]=0;tbar[k]=0;}
#endif
        size_t i=(size_t)blockIdx.x*QSB_RECOVERY_N+threadIdx.x,s=(size_t)n;
#if QSB_PREP_STORE_PTR
        /* The four saved planes are consecutive `n`-record blocks of one
         * allocation, so plane k of this lane is k strides past plane 0.
         * Stepping one pointer by that stride writes the same four addresses
         * the k*s+i index expressions name, with the scaled stride formed
         * once; the stored words and their order are unchanged. */
        ulonglong2 *sp=saved+i;
#if QSB_STREAM2
        qsb_st_v2(sp,vbar[0],vbar[1]);        sp+=s;
        qsb_st_v2(sp,vbar[2],vbar[3]);        sp+=s;
        qsb_st_v2(sp,tbar[0],tbar[1]);        sp+=s;
        qsb_st_v2(sp,tbar[2],tbar[3]);
#else
        *sp=make_ulonglong2(vbar[0],vbar[1]); sp+=s;
        *sp=make_ulonglong2(vbar[2],vbar[3]); sp+=s;
        *sp=make_ulonglong2(tbar[0],tbar[1]); sp+=s;
        *sp=make_ulonglong2(tbar[2],tbar[3]);
#endif
#else
#if QSB_STREAM2
        qsb_st_v2(&saved[0*s+i],vbar[0],vbar[1]);
        qsb_st_v2(&saved[1*s+i],vbar[2],vbar[3]);
        qsb_st_v2(&saved[2*s+i],tbar[0],tbar[1]);
        qsb_st_v2(&saved[3*s+i],tbar[2],tbar[3]);
#else
        saved[0*s+i]=make_ulonglong2(vbar[0],vbar[1]);
        saved[1*s+i]=make_ulonglong2(vbar[2],vbar[3]);
        saved[2*s+i]=make_ulonglong2(tbar[0],tbar[1]);
        saved[3*s+i]=make_ulonglong2(tbar[2],tbar[3]);
#endif
#endif
    }
}

__device__ __forceinline__ uint32_t qsb_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    const uint64_t *a,const uint64_t *b,const uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],s[4];
#if QSB_LAZY_REC
    /* u, v, l, m and sum only feed multiplies and borrow-corrected subtractions, which
     * accept any representative in [0,2^256); only x1/x2 (hashed) and the parity inputs
     * need [0,p). So u and v stay raw and m, sum use the carry-folding lazy add
     * (congruent, [0,2^256); a second carry needs a 2^-223 input, as in the chain). */
    qsb_packed_raw_mul(u,tbar,weighted_inv);
    qsb_packed_raw_mul(v,vbar,root_inv);
#if QSB_REC_SUM2U
    /* l + m = (u - v) + (u + v) = 2u exactly in Z, hence mod p, and `sum` is
     * consumed only by the two slope multiplies, which accept any
     * representative in [0,2^256). Doubling u instead of adding the two
     * slopes gives the same residue with the same lazy carry fold (identical
     * input domain: u and v are both raw 256-bit products), and it lifts the
     * sum off the v multiply and the two slope additions, so the long
     * sum*(l-c) and sum*(m-c) products can start as soon as u exists. */
    _ModSub256(l,u,v); _ModAddLazy(m,u,v); _ModAddLazy(sum,u,u);
#else
    _ModSub256(l,u,v); _ModAddLazy(m,u,v); _ModAddLazy(sum,l,m);
#endif
#else
    qsb_recovery_mul(u,tbar,weighted_inv);
    qsb_recovery_mul(v,vbar,root_inv);
    _ModSub256(l,u,v); _ModAdd256(m,u,v); _ModAdd256(sum,l,m);
#endif
#if QSB_RAW_X
    /* P7: x1, x2 stay raw; raw + a < 2p whenever a[3] != 2^64-1, so the one conditional
     * subtraction in _ModAdd256 still yields canonical x (qsb_add_boundary keeps the
     * normalisation for the other case). */
    _ModSub256(t,l,c); qsb_packed_raw_mul(x1,sum,t); qsb_add_boundary(x1,a); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_packed_raw_mul(x2,sum,t); qsb_add_boundary(x2,a); _ModAdd256(x2,x2,a);
#elif !QSB_PARITY_SUM
    _ModSub256(t,l,c); qsb_recovery_mul(x1,sum,t); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_recovery_mul(x2,sum,t); _ModAdd256(x2,x2,a);
#endif
#if QSB_PARITY_SUM
    /* P9: r_i = x_i - a is the canonical product before "+a" (re-derived here with one
     * subtraction-free identity: x_i - a == sum*(l or m - c)), so a - x_i == -r_i and
     * s1 = l*(a-x1) == -(l*r1), s2 = m*(a-x2) == -(m*r2). See qsb_sum_parity. */
#if QSB_REC_RAW_S
    /* The product only has to be canonical where it is hashed. s enters one
     * _ModAdd256 and one full-width multiply: the multiply takes any 256-bit
     * representative, and qsb_add_boundary is exactly the P7 precondition for
     * the add (raw < 2^256 and a < 2^256-2K give raw+a < 2p, so the single
     * conditional subtraction of _ModAdd256 lands canonical; the a[3]==2^64-1
     * tail keeps the normalisation). The boundary call rewrites s only in that
     * tail and then to a congruent value, so l*s and m*s are unchanged mod p
     * and the two parities are the ones the canonical product produced. */
    _ModSub256(t,l,c); qsb_packed_raw_mul(s,sum,t); qsb_add_boundary(s,a); _ModAdd256(x1,s,a);
    qsb_packed_raw_mul(u,l,s);
    _ModSub256(t,m,c); qsb_packed_raw_mul(s,sum,t); qsb_add_boundary(s,a); _ModAdd256(x2,s,a);
    qsb_packed_raw_mul(v,m,s);
#else
    _ModSub256(t,l,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x1,s,a);
    qsb_packed_raw_mul(u,l,s);
    _ModSub256(t,m,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x2,s,a);
    qsb_packed_raw_mul(v,m,s);
#endif
    return qsb_sum_parity(u,b,1u)|(qsb_sum_parity(v,b,0u)<<1);
}
#else
    _ModSub256(t,a,x1); qsb_packed_raw_mul(s,l,t); qsb_parity_boundary(s,b);
    uint32_t parity=qsb_difference_parity(s,b);
    _ModSub256(t,a,x2); qsb_packed_raw_mul(s,m,t); qsb_parity_boundary(s,b);
    return parity|(qsb_difference_parity(b,s)<<1);
}
#endif

#ifndef QSB_S2_SPLIT_RECOVER
#define QSB_S2_SPLIT_RECOVER 1 /* the two public keys are recovered one at a time */
#endif
#if QSB_S2_SPLIT_RECOVER
#if !(QSB_LAZY_REC && QSB_PARITY_SUM && QSB_REC_RAW_S && QSB_REC_SUM2U)
#error "the split recovery reproduces the lazy sum-parity finish"
#endif
/* Same identities as qsb_packed_finish, cut where its two halves stop
 * sharing values. Everything both public keys need -- the two inverse
 * products and the three slope words -- is formed once, exactly as before;
 * what remains per key is one difference, the abscissa product with its "+a"
 * boundary and the parity product. The two keys are then never live at the
 * same time: the second key's abscissa and parity are produced after the
 * first key's hash and gate, so the four abscissa words, the parity product
 * and the second slope's difference no longer sit in registers across a
 * sixty-four-round compression. Identical operands, identical order within a
 * key, and the two gate tests keep their order and their hit encoding. */
__device__ __forceinline__ void qsb_packed_slopes(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,uint64_t *l,uint64_t *m,uint64_t *sum) {
    uint64_t u[4],v[4];
    qsb_packed_raw_mul(u,tbar,weighted_inv);
    qsb_packed_raw_mul(v,vbar,root_inv);
    _ModSub256(l,u,v); _ModAddLazy(m,u,v); _ModAddLazy(sum,u,u);
}
__device__ __forceinline__ uint32_t qsb_packed_branch(
    const uint64_t *slope,const uint64_t *sum,const uint64_t *a,const uint64_t *b,
    const uint64_t *c,uint32_t neg,uint64_t *x) {
    uint64_t t[4],s[4],w[4];
    _ModSub256(t,slope,c); qsb_packed_raw_mul(s,sum,t); qsb_add_boundary(s,a);
    _ModAdd256(x,s,a);
    qsb_packed_raw_mul(w,slope,s);
    return qsb_sum_parity(w,b,neg);
}
#endif
