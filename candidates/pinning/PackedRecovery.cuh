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
// Shared-multiplicand dual product: out0=a0*s, out1=a1*s. Recovery leaves
// vbar=Y*hc and tbar=V*hc share hc and are otherwise independent, so one
// helper issues both exact muls before any mask or global store.
__device__ __forceinline__ void qsb_packed_raw_mul(
    uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];qsb_field_mul(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    Load256(out,tmp);
}
__device__ __forceinline__ void qsb_packed_raw_mul2_shared(
    uint64_t *out0,const uint64_t *a0,uint64_t *out1,const uint64_t *a1,
    const uint64_t *s) {
    uint64_t t0[5],t1[5];
    qsb_field_mul(t0,const_cast<uint64_t*>(a0),const_cast<uint64_t*>(s));
    qsb_field_mul(t1,const_cast<uint64_t*>(a1),const_cast<uint64_t*>(s));
    Load256(out0,t0);
    Load256(out1,t1);
}
__device__ __forceinline__ void qsb_packed_raw_mul2(
    uint64_t *out0,const uint64_t *a0,const uint64_t *b0,
    uint64_t *out1,const uint64_t *a1,const uint64_t *b1) {
    uint64_t t0[5],t1[5];
    qsb_field_mul(t0,const_cast<uint64_t*>(a0),const_cast<uint64_t*>(b0));
    qsb_field_mul(t1,const_cast<uint64_t*>(a1),const_cast<uint64_t*>(b1));
    Load256(out0,t0);
    Load256(out1,t1);
}
__device__ __forceinline__ void qsb_recovery_mul2(
    uint64_t *out0,const uint64_t *a0,const uint64_t *b0,
    uint64_t *out1,const uint64_t *a1,const uint64_t *b1) {
    uint64_t t0[5],t1[5];
    qsb_field_mul(t0,const_cast<uint64_t*>(a0),const_cast<uint64_t*>(b0));
    qsb_field_mul(t1,const_cast<uint64_t*>(a1),const_cast<uint64_t*>(b1));
    qsb_field_normalize(t0);
    qsb_field_normalize(t1);
    Load256(out0,t0);
    Load256(out1,t1);
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
        qsb_packed_raw_mul2_shared(vbar,Y,tbar,V,hc);
        if(!usable)for(int k=0;k<4;k++){vbar[k]=0;tbar[k]=0;}
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
    uint64_t u[4],v[4],l[4],m[4],sum[4],t1[4],t2[4],s1[4],s2[4];
    qsb_recovery_mul2(u,tbar,weighted_inv,v,vbar,root_inv);
    _ModSub256(l,u,v); _ModAdd256(m,u,v); _ModAdd256(sum,l,m);
    _ModSub256(t1,l,c); _ModSub256(t2,m,c);
    qsb_packed_raw_mul2_shared(x1,t1,x2,t2,sum);
    qsb_add_boundary(x1,a); _ModAdd256(x1,x1,a);
    qsb_add_boundary(x2,a); _ModAdd256(x2,x2,a);
    _ModSub256(t1,a,x1); _ModSub256(t2,a,x2);
    qsb_packed_raw_mul2(s1,l,t1,s2,m,t2);
    qsb_parity_boundary(s1,b); qsb_parity_boundary(s2,b);
    uint32_t parity=qsb_difference_parity(s1,b);
    return parity|(qsb_difference_parity(b,s2)<<1);
}
