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
    // All lanes finish reading their digits/anchor before tree overwrites.
    __syncthreads();
    uint64_t (*products)[2*QSB_RECOVERY_N]=(uint64_t (*)[2*QSB_RECOVERY_N])qsb_digit_arena();
    uint64_t (*excluded)[QSB_RECOVERY_N]=(uint64_t (*)[QSB_RECOVERY_N])(qsb_digit_arena()+8*QSB_TREE_N);
    qsb_cofactor_prepare<QSB_RECOVERY_N>(D,roots,products,excluded);
    if(active) {
        uint64_t hc[4],vbar[4],tbar[4];
        qsb_packed_raw_mul(hc,U,D);
#ifndef QSB_MASK_HC
#define QSB_MASK_HC 1
#endif
#if QSB_MASK_HC
        if(!usable)for(int k=0;k<4;k++)hc[k]=0;
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
#else
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
        if(!usable)for(int k=0;k<4;k++){vbar[k]=0;tbar[k]=0;}
#endif
        size_t i=(size_t)blockIdx.x*QSB_RECOVERY_N+threadIdx.x,s=(size_t)n;
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
    }
}

__device__ __forceinline__ uint32_t qsb_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],s[4];
#if QSB_LAZY_REC
    /* u, v, l, m and sum only feed multiplies and borrow-corrected subtractions, which
     * accept any representative in [0,2^256); only x1/x2 (hashed) and the parity inputs
     * need [0,p). So u and v stay raw and m, sum use the carry-folding lazy add
     * (congruent, [0,2^256); a second carry needs a 2^-223 input, as in the chain). */
    qsb_packed_raw_mul(u,tbar,weighted_inv);
    qsb_packed_raw_mul(v,vbar,root_inv);
    _ModSub256(l,u,v); _ModAddLazy(m,u,v); _ModAddLazy(sum,l,m);
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
#ifndef QSB_LAZY_PARITY_X
#define QSB_LAZY_PARITY_X 1
#endif
    /* P9: r_i = x_i - a is the canonical product before "+a" (re-derived here with one
     * subtraction-free identity: x_i - a == sum*(l or m - c)), so a - x_i == -r_i and
     * s1 = l*(a-x1) == -(l*r1), s2 = m*(a-x2) == -(m*r2). See qsb_sum_parity. */
#if QSB_LAZY_PARITY_X
    _ModSub256(t,l,c); qsb_packed_raw_mul(s,sum,t); qsb_packed_raw_mul(u,l,s);
    qsb_add_boundary(s,a); _ModAdd256(x1,s,a);
    _ModSub256(t,m,c); qsb_packed_raw_mul(s,sum,t); qsb_packed_raw_mul(v,m,s);
    qsb_add_boundary(s,a); _ModAdd256(x2,s,a);
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
