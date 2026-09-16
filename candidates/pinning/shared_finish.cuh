#pragma once

__device__ __forceinline__ bool qsb_finish_zero(const uint64_t *a) {
    return (a[0]|a[1]|a[2]|a[3])==0 ||
        (a[0]==0xFFFFFFFEFFFFFC2FULL && a[1]==UINT64_MAX &&
         a[2]==UINT64_MAX && a[3]==UINT64_MAX);
}

__device__ __forceinline__ bool qsb_recover_pair_shared(
    uint64_t *X,uint64_t *Y,uint64_t *Z,uint64_t *rx,uint64_t *ry,
    uint64_t *x1,uint64_t *y1,uint64_t *x2,uint64_t *y2) {
    if(qsb_finish_zero(Z)) return false;
    uint64_t d[4],w[5]={0,0,0,0,0};
    _ModMult(d,rx,Z);
    _ModSub256(d,d,X);
    if(qsb_finish_zero(d)) return false;
    _ModMult(w,Z,d);
    _ModInv(w);

    uint64_t iz[4],id[4],px[4],py[4],ryz[4],m1[4],m2[4],sumx[4],t[4];
    _ModMult(iz,d,w);
    _ModMult(id,Z,w);
    _ModMult(px,X,iz);
    _ModMult(py,Y,iz);
    _ModMult(ryz,ry,Z);
    _ModSub256(m1,ryz,Y);
    _ModMult(m1,m1,id);
    _ModAdd256(m2,ryz,Y);
    _ModMult(m2,m2,id);
    _ModAdd256(sumx,px,rx);

    _ModSqr(x1,m1);
    _ModSub256(x1,x1,sumx);
    _ModSub256(t,px,x1);
    _ModMult(y1,m1,t);
    _ModSub256(y1,y1,py);

    _ModSqr(x2,m2);
    _ModSub256(x2,x2,sumx);
    _ModSub256(t,x2,px);
    _ModMult(y2,m2,t);
    _ModSub256(y2,y2,py);
    return true;
}
