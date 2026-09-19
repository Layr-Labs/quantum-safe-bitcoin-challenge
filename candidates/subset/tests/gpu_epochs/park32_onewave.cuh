// GPL-3.0-only; retains the inherited VanitySearch and tree authorship.
// Park32 one-wave integration of the user-supplied 2026-09-19 research.
// TREE1/TREE2 bodies below are caller-owned transcriptions of promoted 9fab500.
#pragma once
#ifndef QSB_PARK32_ONEWAVE
#define QSB_PARK32_ONEWAVE 1
#endif
#if QSB_PARK32_ONEWAVE
#if !QSB_PAIR_SHARED || !ZLAB_K2S3M || ZLAB_T14 || !ZLAB_DIRDIG || ZLAB_TREE != 2
#error "park32 requires the ranked paired 15-term direct-digit TREE2/3M configuration"
#endif
union QsbOWShared {
    struct { uint64_t park[16][256]; uint64_t tree[4][512]; } fixed;
    struct { uint64_t park[12][256]; uint64_t products[4][512]; uint64_t inverses[4][256]; } recovery;
};
static_assert(sizeof(QsbOWShared)==49152, "phase overlay must be 48 KiB");
__device__ __forceinline__ void qsb_ow_tree1(uint64_t *value, uint64_t (&tree)[4][512]){
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
    __syncthreads();
    // Upward levels with at least two writers; the readers of level `width`
    // are its writers' low half, so a warp barrier suffices once width<=32.
    #pragma unroll 1
    for(int width=n>>1;width>1;width>>=1){
        if(tid<width){
            int node=width+tid;
            uint64_t a[5],b[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        if(width>32)__syncthreads();else __syncwarp();
    }
    // Keep the promoted cooperative root inverse in the compact heap arena.
#if HM43_WARP_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;zi_inverse_quad(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][2+tid]=child[k];
        }
    }
#elif HM41_QUAD_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm41_quad_inverse(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][2+tid]=child[k];
        }
    }
#elif HM39_PAIR_ROOT
    if(tid<2){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
        a[4]=b[4]=0;__syncwarp(0x00000003);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm39_pair_inverse(root,tid);
        uint64_t child[5];
        #pragma unroll
        for(int k=0;k<4;k++)child[k]=tid? a[k]:b[k];
        child[4]=0;qsb_field_mul_raw(child,root,child);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][2+tid]=child[k];
    }
#else
    if(tid==0){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4]=0;
        qsb_field_mul_raw(a,root,a);   /* 1/b */
        qsb_field_mul_raw(b,root,b);   /* 1/a */
        // inverse index = product index - n
        #pragma unroll
        for(int k=0;k<4;k++){tree[k][2]=b[k];tree[k][3]=a[k];}
    }
#endif
    __syncwarp();
    // Remaining internal downward levels: level `width` is read by lanes < 2*width.
    #pragma unroll 1
    for(int width=2;width<(n>>1);width<<=1){
        if(tid<width){
            int node=width+tid;
            uint64_t parent[5],left[5],right[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];
                left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            parent[4]=left[4]=right[4]=0;
            qsb_field_mul_raw(right,parent,right);qsb_field_mul_raw(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
            }
        }
        // Level `width` is read by lanes < 2*width, except the last internal
        // level (width == n/4), which every lane reads for its own leaf.
        if((width<<1)>32 || (width<<2)==n)__syncthreads();else __syncwarp();
    }
    // Leaf level: every lane multiplies its parent inverse by its sibling's
    // (never overwritten) leaf product. No shared write, no barrier.
    {
        uint64_t parent[5],sibling[5];
        const int leaf=n+tid;
        #pragma unroll
        for(int k=0;k<4;k++){parent[k]=tree[k][leaf>>1];sibling[k]=tree[k][leaf^1];}
        parent[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent,sibling);
    }
    value[4]=0;
}
__device__ __forceinline__ void qsb_ow_tree2(uint64_t *value, uint64_t (&products)[4][512], uint64_t (&inverses)[4][256]){
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    // Level (offset,count): (0,n),(n,n/2),...,(2n-4,2). Level `count` is
    // formed by lanes < count/2 and read by lanes < count/4.
    int offset=0;
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    // offset == 2n-4: the two root children.
#if HM43_WARP_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;zi_inverse_quad(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
    }
#elif HM41_QUAD_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm41_quad_inverse(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
    }
#elif HM39_PAIR_ROOT
    if(tid<2){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x00000003);qsb_field_mul_raw(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm39_pair_inverse(root,tid);
        uint64_t child[5];
        #pragma unroll
        for(int k=0;k<4;k++)child[k]=tid? a[k]:b[k];
        child[4]=0;qsb_field_mul_raw(child,root,child);
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
    }
#else
    if(tid==0){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4]=0;
        qsb_field_mul_raw(a,root,a);   /* 1/b */
        qsb_field_mul_raw(b,root,b);   /* 1/a */
        // inverse index = product index - n
        #pragma unroll
        for(int k=0;k<4;k++){inverses[k][offset-n]=b[k];inverses[k][offset-n+1]=a[k];}
    }
#endif
    __syncwarp();
    // Level count (4..n/2): lanes < count read parent inverses written by
    // lanes < count/2 and write inverses read by lanes < 2*count.
    offset-=4;   /* level count=4 */
    #pragma unroll 1
    for(int count=4;count<n;count<<=1){
        int half=count>>1;
        if(tid<count){
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-n+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul_raw(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child_inv[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    // offset == 0 would be the leaf level; lanes form their own leaf inverse.
    {
        const int half=n>>1;
        uint64_t parent_inv[5],sibling[5];
        #pragma unroll
        for(int k=0;k<4;k++){
            parent_inv[k]=inverses[k][tid&(half-1)];
            sibling[k]=products[k][tid^half];
        }
        parent_inv[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent_inv,sibling);
    }
    value[4]=0;
}

// Use a function, not GPUMath's compound-statement Load256 macro: the
// latter followed by a semicolon breaks unbraced if/else control flow.
__device__ __forceinline__ void qsb_ow_copy(uint64_t *r,const uint64_t *a) {
    #pragma unroll
    for(int k=0;k<4;k++)r[k]=a[k];
}
__device__ __forceinline__ bool qsb_ow_zero(uint64_t *x) {
    qsb_field_normalize(x);
    return !(x[0]|x[1]|x[2]|x[3]);
}
__device__ __forceinline__ void qsb_ow_one(uint64_t *x) {
    x[0]=1; x[1]=x[2]=x[3]=0;
}
__device__ __forceinline__ void qsb_ow_save(uint64_t (*park)[256],int row,const uint64_t *x) {
    #pragma unroll
    for(int k=0;k<4;k++)park[row+k][threadIdx.x]=x[k];
}
__device__ __forceinline__ void qsb_ow_load(uint64_t *x,const uint64_t (*park)[256],int row) {
    #pragma unroll
    for(int k=0;k<4;k++)x[k]=park[row+k][threadIdx.x];
}
__device__ __forceinline__ void qsb_ow_x(const uint8_t *table,const uint64_t *M,int c,uint64_t *x) {
    uint32_t idx; uint64_t ignored;
    gt_direct_digit(M,0,(unsigned)gt_shift(c)+1u,gt_width(c),c==14,&idx,&ignored);
    const ulonglong2 *p=(const ulonglong2 *)(table+((size_t)gt_offset(c)+idx)*64u);
    const ulonglong2 a=__ldg(p),b=__ldg(p+1);
    x[0]=a.x;x[1]=a.y;x[2]=b.x;x[3]=b.y;
}
__device__ __forceinline__ void qsb_ow_point(const uint8_t *table,const uint64_t *M,uint64_t sign,int c,uint64_t *x,uint64_t *y) {
    uint32_t idx;uint64_t neg;
    gt_direct_digit(M,sign,(unsigned)gt_shift(c)+1u,gt_width(c),c==14,&idx,&neg);
    gt_load_signed_flat(table,gt_offset(c),idx,neg,x,y);
}
__device__ __forceinline__ void qsb_ow_den(const uint8_t *table,const uint64_t *M,int pair,uint64_t *d) {
    uint64_t a[4],b[4];
    qsb_ow_x(table,M,2*pair,a);qsb_ow_x(table,M,2*pair+1,b);
    _ModSub256(d,b,a);qsb_field_normalize(d);
}
// SHA input construction is exactly the promoted qsb_k2s_front3 prefix.
__device__ __forceinline__ void qsb_ow_scalar(const epoch_desc_t *ep,const uint32_t *first,int lane,uint64_t *z) {
    uint32_t state[8];
    #pragma unroll
    for(int i=0;i<8;i++)state[i]=ep->mid[i];
    qsb_scheduled_window_hash(state,ep,lane,first);
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++)b2[i]=state[i];
    b2[8]=0x80000000;
    #pragma unroll
    for(int i=9;i<15;i++)b2[i]=0;
    b2[15]=0x100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);
    #pragma unroll
    for(int i=0;i<4;i++)z[i]=((uint64_t)s2[6-2*i]<<32)|s2[7-2*i];
}
// A owns shared scalar/P1/P4. B owns shared P3 and retains scalar/P1/P5.
// Replacing a zero candidate's entire product by one prevents cross-lane poisoning.
template<bool B>
__device__ __forceinline__ bool qsb_ow_prefix(const uint64_t *z,const uint8_t *table,uint64_t *total,uint64_t *p1,uint64_t *p45,uint64_t (*park)[256]) {
    uint64_t M[4];int sign;gt_recode_setup(z,M,&sign);
    uint64_t d[4];bool bad=false;
    qsb_ow_one(total);
    #pragma unroll 1
    for(int j=0;j<7;j++) {
        qsb_ow_den(table,M,j,d);
        if(qsb_ow_zero(d)){bad=true;qsb_ow_one(d);}
        if(j==0)qsb_ow_copy(total,d);else _ModMult(total,d);
        if(j==1)qsb_ow_copy(p1,total);
        if(j==(B?5:4))qsb_ow_copy(p45,total);
        if(B && j==3)qsb_ow_save(park,12,total);
    }
    if(bad)qsb_ow_one(total);
    return bad;
}
struct QsbOWA {uint64_t product[4];int bad;};
struct QsbOWB {uint64_t scalar[4],p1[4],p5[4],product[4];int bad;};
__device__ __noinline__ QsbOWA qsb_ow_prepare_a(const epoch_desc_t *ep,const uint32_t *first,int lane,const uint8_t *table,uint64_t (*park)[256]) {
    uint64_t z[4],p1[4],p4[4];QsbOWA out;
    qsb_ow_scalar(ep,first,lane,z);qsb_ow_save(park,0,z);
    out.bad=qsb_ow_prefix<false>(z,table,out.product,p1,p4,park);
    qsb_ow_save(park,4,p1);qsb_ow_save(park,8,p4);return out;
}
__device__ __noinline__ QsbOWB qsb_ow_prepare_b(const epoch_desc_t *ep,const uint32_t *first,int lane,const uint8_t *table,uint64_t (*park)[256]) {
    QsbOWB out;qsb_ow_scalar(ep,first,lane,out.scalar);
    out.bad=qsb_ow_prefix<true>(out.scalar,table,out.product,out.p1,out.p5,park);return out;
}

// Construct one affine pair with a supplied inverse of its x difference.
// d is already available from the full loads, so the reverse prefix can advance.
__device__ __forceinline__ void qsb_ow_affine(uint64_t *x,uint64_t *y,
    uint64_t *a,uint64_t *ay,uint64_t *b,uint64_t *by,uint64_t *inverse) {
    uint64_t m[4],t[4];
    _ModSub256(m,by,ay);_ModMult(m,inverse);
    _ModSqr(t,m);_ModSub256(t,t,a);_ModSub256(x,t,b);
    _ModSub256(t,a,x);_ModMult(t,m);_ModSub256(y,t,ay);
}
// Reordered tail uses the crown's speculative filter and its unchanged exact
// hit verifier. Any zero projective denominator requests the original chain.
__device__ __forceinline__ void qsb_ow_append(int j,uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    uint64_t *anchor,uint64_t *px,uint64_t *py,uint32_t &bad) {
    if(j==6){qsb_ow_copy(X,px);qsb_ow_copy(Y,py);qsb_ow_copy(anchor,py);return;}
    if(j==5){
        uint64_t ax[4],ay[4];qsb_ow_copy(ax,X);qsb_ow_copy(ay,Y);
        qsb_filter_point_seed(X,Y,ZZ,ZZZ,ax,ay,px,py,bad);
        // Seed's deferred Y is anchored to its FIRST affine point.
    }else{
        qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,px,py,anchor,bad);
        qsb_ow_copy(anchor,py);
    }
    if(qsb_ow_zero(ZZ))bad|=1;
}

template<bool B>
__device__ __forceinline__ void qsb_ow_sum(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t *scalar,const uint8_t *table,uint64_t *inverse,
    uint64_t *p1,uint64_t *p45,bool failed,uint64_t *joint_inverse,QsbOWShared &shared) {
    if(failed){
        if(!B){
            #pragma unroll
            for(int k=0;k<4;k++)shared.fixed.tree[k][threadIdx.x]=joint_inverse[k];
        }
        _FixedBaseSignedXYZZStream(X,Y,ZZ,ZZZ,scalar,table);
        return;
    }
    uint64_t M[4];int sign;gt_recode_setup(scalar,M,&sign);
    uint64_t p2[4],anchor[4];uint32_t bad=0;
    #pragma unroll 1
    for(int j=6;j>=0;j--){
        uint64_t prefix[4],d[4];
        if(j==6){
            if(B)qsb_ow_copy(prefix,p45);
            else {qsb_ow_den(table,M,5,d);_ModMult(prefix,p45,d);}
        }else if(j==5){
            if(B){qsb_ow_load(prefix,shared.fixed.park,12);qsb_ow_den(table,M,4,d);_ModMult(prefix,d);}
            else qsb_ow_copy(prefix,p45);
        }else if(j==4){
            if(B)qsb_ow_load(prefix,shared.fixed.park,12);
            else {qsb_ow_den(table,M,2,d);_ModMult(p2,p1,d);qsb_ow_den(table,M,3,d);_ModMult(prefix,p2,d);}
        }else if(j==3){
            if(B){qsb_ow_den(table,M,2,d);_ModMult(prefix,p1,d);}
            else qsb_ow_copy(prefix,p2);
        }else if(j==2)qsb_ow_copy(prefix,p1);
        else if(j==1){
            // One explicit x-only d0 reread keeps four pair-0 coordinates out
            // of the live set. This is +64 logical table bytes vs the paper.
            qsb_ow_den(table,M,0,prefix);
        }
        uint64_t a[4],ay[4],b[4],by[4],di[4],px[4],py[4];
        qsb_ow_point(table,M,(uint64_t)(sign<0),2*j,a,ay);
        qsb_ow_point(table,M,(uint64_t)(sign<0),2*j+1,b,by);
        _ModSub256(d,b,a);
        if(!B && j==6){
            // PA = P5*d6. Store inverse(PB) in the now-dead TREE1 arena;
            // the caller's post-tree barrier has retired all sibling reads.
            uint64_t total[4],ib[5];_ModMult(total,prefix,d);
            qsb_field_mul_raw(ib,joint_inverse,total);
            #pragma unroll
            for(int k=0;k<4;k++)shared.fixed.tree[k][threadIdx.x]=ib[k];
        }
        if(j){_ModMult(di,inverse,prefix);_ModMult(inverse,d);}
        else qsb_ow_copy(di,inverse);
        qsb_ow_affine(px,py,a,ay,b,by,di);
        qsb_ow_append(j,X,Y,ZZ,ZZZ,anchor,px,py,bad);
    }
    {
        uint64_t x[4],y[4];qsb_ow_point(table,M,(uint64_t)(sign<0),14,x,y);
        // No special final doubling can accidentally turn an earlier zero-ZZ
        // into a valid-looking point; replay handles all singular chains.
        qsb_filter_point_add<false>(X,Y,ZZ,ZZZ,x,y,anchor,bad);
        if(qsb_ow_zero(ZZ))bad|=1;
    }
    if(bad)_FixedBaseSignedXYZZStream(X,Y,ZZ,ZZZ,scalar,table);
}
__device__ __forceinline__ QsbPairFront3 qsb_ow_recovery(uint64_t *x,uint64_t *y,uint64_t *zz,uint64_t *zzz) {
    uint64_t rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
    uint64_t ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};
    uint64_t prod[5];QsbPairFront3 out;
    qsb_xyzz_finish_prepare(x,zz,zzz,rx,prod);
    qsb_k2s_pre3(y,zz,zzz,ry,out.words+4);
    out.ok=!qsb_ow_zero(prod);qsb_ow_copy(out.words,prod);return out;
}
__device__ __noinline__ QsbPairFront3 qsb_ow_finish_a(const uint8_t *table,uint64_t *joint,uint64_t *pb,bool failed,QsbOWShared &shared) {
    uint64_t z[4],p1[4],p4[4],inv[5],x[4],y[4],zz[4],zzz[4];
    qsb_ow_load(z,shared.fixed.park,0);qsb_ow_load(p1,shared.fixed.park,4);qsb_ow_load(p4,shared.fixed.park,8);
    qsb_field_mul_raw(inv,joint,pb);
    qsb_ow_sum<false>(x,y,zz,zzz,z,table,inv,p1,p4,failed,joint,shared);
    return qsb_ow_recovery(x,y,zz,zzz);
}
__device__ __noinline__ QsbPairFront3 qsb_ow_finish_b(const uint8_t *table,QsbOWB &b,QsbOWShared &shared) {
    uint64_t inv[5],x[4],y[4],zz[4],zzz[4];
    #pragma unroll
    for(int k=0;k<4;k++)inv[k]=shared.fixed.tree[k][threadIdx.x];
    inv[4]=0;
    qsb_ow_sum<true>(x,y,zz,zzz,b.scalar,table,inv,b.p1,b.p5,b.bad,inv,shared);
    return qsb_ow_recovery(x,y,zz,zzz);
}
#endif // QSB_PARK32_ONEWAVE
