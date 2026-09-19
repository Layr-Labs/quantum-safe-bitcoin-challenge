#pragma once
__device__ __forceinline__ void qsb_field_mul(uint64_t*,uint64_t*,uint64_t*);
__device__ __forceinline__ bool q9_zero(const uint64_t *p){
    return !(p[0]|p[1]|p[2]|p[3]) ||
        ((p[1]&p[2]&p[3])==0xffffffffffffffffULL&&p[0]==0xfffffffefffffc2fULL);
}
__device__ __forceinline__ void q9_special_add(uint64_t*X,uint64_t*Y,uint64_t*U,uint64_t*V,
    const uint64_t*ax,const uint64_t*ay,const uint64_t*anchor,const uint64_t*R){
    if(q9_zero(U)){
        Load256(X,ax);_ModAdd256(Y,ay,ay);U[0]=V[0]=1;U[1]=U[2]=U[3]=V[1]=V[2]=V[3]=0;return;
    }
    if(!q9_zero(R)){
        #pragma unroll
        for(int j=0;j<4;j++)X[j]=Y[j]=U[j]=V[j]=0;return;
    }
    uint64_t yy[4],a[4],b[4],c[4],d[4],m[4],nx[4],ny[4],t[4];
    qsb_field_mul(t,(uint64_t*)anchor,V);_ModSub256(yy,Y,t);
    _ModAdd256(a,yy,yy);qsb_field_mul(b,a,a);qsb_field_mul(c,a,b);qsb_field_mul(d,X,b);
    qsb_field_mul(m,X,X);_ModAdd256(t,m,m);_ModAdd256(m,t,m);
    qsb_field_mul(nx,m,m);_ModSub256(nx,nx,d);_ModSub256(nx,nx,d);
    _ModSub256(t,d,nx);qsb_field_mul(ny,m,t);qsb_field_mul(t,c,yy);_ModSub256(ny,ny,t);
    qsb_field_mul(U,U,b);qsb_field_mul(V,V,c);
    qsb_field_mul(t,(uint64_t*)ay,V);_ModAdd256(Y,ny,t);Load256(X,nx);
}
