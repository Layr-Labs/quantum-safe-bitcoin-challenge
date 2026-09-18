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

// Exact full-width residue; callers normalize before additions/parity.
__device__ __forceinline__ void qsb_packed_raw_mul(
    uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];qsb_field_mul(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    Load256(out,tmp);
}

// Combine the public cofactor traversal with our existing exact/canonical
// recovery boundary and the odinfree square-free finish identity.
// Independent leaves issue through distinct product buffers: tbar/vbar share
// only hc, u/v share nothing, and the x-pair shares only S.
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
        // Weighted leaf tbar=V*hc is issued first (finish consumes u=tbar*J
        // first). The sibling vbar=Y*hc is independent and uses its own
        // product buffer so it is not stuck behind tbar's store.
        uint64_t ttmp[5],vtmp[5];
        qsb_field_mul(ttmp,const_cast<uint64_t*>(V),hc);
        qsb_field_mul(vtmp,const_cast<uint64_t*>(Y),hc);
        Load256(tbar,ttmp);
        Load256(vbar,vtmp);
        if(!usable)for(int k=0;k<4;k++){vbar[k]=0;tbar[k]=0;}
        size_t i=(size_t)blockIdx.x*QSB_RECOVERY_N+threadIdx.x,s=(size_t)n;
        saved[0*s+i]=make_ulonglong2(vbar[0],vbar[1]);
        saved[1*s+i]=make_ulonglong2(vbar[2],vbar[3]);
        saved[2*s+i]=make_ulonglong2(tbar[0],tbar[1]);
        saved[3*s+i]=make_ulonglong2(tbar[2],tbar[3]);
    }
}

__device__ __forceinline__ uint32_t qsb_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t1[4],t2[4],s[4];
    // Dual leaf muls: u=tbar*J and v=vbar*I share no product buffer.
    // Canonicalize both after both products so the second mul is not stuck
    // behind the first qsb_recovery_mul's normalize. Same residues.
    uint64_t ut[5],vt[5];
    qsb_field_mul(ut,const_cast<uint64_t*>(tbar),const_cast<uint64_t*>(weighted_inv));
    qsb_field_mul(vt,const_cast<uint64_t*>(vbar),const_cast<uint64_t*>(root_inv));
    qsb_field_normalize(ut); qsb_field_normalize(vt);
    Load256(u,ut); Load256(v,vt);
    _ModSub256(l,u,v); _ModAdd256(m,u,v); _ModAdd256(sum,l,m);
    // Independent x-pair leaves share sum but not the (·-c) factor. Split the
    // reused t so both multiplies issue before either +a. Reuse ut/vt.
    _ModSub256(t1,l,c); _ModSub256(t2,m,c);
    qsb_field_mul(ut,sum,t1);
    qsb_field_mul(vt,sum,t2);
    qsb_field_normalize(ut); qsb_field_normalize(vt);
    Load256(x1,ut); Load256(x2,vt);
    _ModAdd256(x1,x1,a); _ModAdd256(x2,x2,a);
    _ModSub256(t1,a,x1); qsb_packed_raw_mul(s,l,t1); qsb_parity_boundary(s,b);
    uint32_t parity=qsb_difference_parity(s,b);
    _ModSub256(t2,a,x2); qsb_packed_raw_mul(s,m,t2); qsb_parity_boundary(s,b);
    return parity|(qsb_difference_parity(b,s)<<1);
}
