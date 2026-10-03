// Paired two-CTA layout: separate immutable products from downward inverses.
// This restores the audited earlier tree layout, retaining current root-entry
// warp synchronization. The extra 8 KiB removes destructive-read barriers.
// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once


#ifndef QSB_TREE_STATIC_N
#define QSB_TREE_STATIC_N 0
#endif



#ifndef QSB_TREE_CONST_ADDR
#define QSB_TREE_CONST_ADDR 0
#endif
#if QSB_TREE_CONST_ADDR != 0 && QSB_TREE_CONST_ADDR != 1
#error "QSB_TREE_CONST_ADDR must be 0 or 1"
#endif





#ifndef QSB_TREE_BOTTOM_FUSE
#define QSB_TREE_BOTTOM_FUSE 0
#endif




#ifndef QSB_TREE_BOTTOM_SHFL
#define QSB_TREE_BOTTOM_SHFL 0
#endif
#if QSB_TREE_BOTTOM_SHFL != 0 && QSB_TREE_BOTTOM_SHFL != 1
#error "QSB_TREE_BOTTOM_SHFL must be 0 or 1"
#endif




#ifndef QSB_TREE_LEAF_SHFL
#define QSB_TREE_LEAF_SHFL 0
#endif
#if QSB_TREE_LEAF_SHFL != 0 && QSB_TREE_LEAF_SHFL != 1
#error "QSB_TREE_LEAF_SHFL must be 0 or 1"
#endif

/* Aligned limb-pair tree rows, adapted from public promoted2f57d80 (GPL-3).
 * CTA extents follow this tree, not the donor's fixed CTA256 storage. */
#ifndef QSB_TREE_ROW128
#define QSB_TREE_ROW128 0
#endif
#if QSB_TREE_ROW128 != 0 && QSB_TREE_ROW128 != 1
#error "QSB_TREE_ROW128 must be 0 or 1"
#endif

#ifndef QSB_TREE_LIVE_MASK
#define QSB_TREE_LIVE_MASK 0
#endif
#if QSB_TREE_LIVE_MASK != 0 && QSB_TREE_LIVE_MASK != 1
#error "QSB_TREE_LIVE_MASK must be0 or1"
#endif
#if QSB_TREE_LIVE_MASK && (!QSB_TREE_WAVE_TOP || QSB_TREE_TOP_SHFL)
#error "QSB_TREE_LIVE_MASK requires shared wave-top, not shuffle experiment"
#endif
#if QSB_TREE_LIVE_MASK
#define QSB_TREE_LIVE_IF(c) if(c)
#else
#define QSB_TREE_LIVE_IF(c)
#endif
#ifndef QSB_ISO_FUSED_ROOT_SCALE
#define QSB_ISO_FUSED_ROOT_SCALE 1
#endif
#ifndef QSB_TREE_TOP_SHFL
#define QSB_TREE_TOP_SHFL 0
#endif
#if QSB_TREE_TOP_SHFL && !QSB_TREE_WAVE_TOP
#error "QSB_TREE_TOP_SHFL requires the wave-top tree"
#endif
#include "hm39_pair_inverse.cuh"
#include "hm41_quad_inverse.cuh"
#include "hm43_warp_inverse.cuh"
#include "zinv32.cuh"
#ifndef QSB_INVERSE_LIMBS
#define QSB_INVERSE_LIMBS 1
#endif
#if QSB_INVERSE_LIMBS
#include "inverse_limbs.cuh"
#endif















#ifndef ZLAB_TREE
#define ZLAB_TREE 2
#endif
#if QSB_TREE_BOTTOM_FUSE && (ZLAB_TREE != 2 || QSB_TREE_ROW128 || QSB_TREE_UNROLL || QSB_ROOT_WARP || QSB_FRONT3_PUBLISH_AB || QSB_SE_BLOCK < 128)
#error "bottom fusion requires the original scalar level-packed tree, CTA >= 128"
#endif
#if QSB_TREE_BOTTOM_SHFL && (QSB_TREE_BOTTOM_FUSE || ZLAB_TREE != 2 || QSB_TREE_ROW128 || QSB_TREE_UNROLL || QSB_ROOT_WARP || QSB_FRONT3_PUBLISH_AB || QSB_SE_BLOCK != 128 || QSB_CTA_WINDOW_SPLIT)
#error "bottom shuffle requires paired scalar CTA128 tree and no bottom fusion"
#endif
#if QSB_TREE_LEAF_SHFL && (QSB_TREE_BOTTOM_SHFL || QSB_TREE_BOTTOM_FUSE || ZLAB_TREE != 2 || QSB_TREE_ROW128 || QSB_TREE_UNROLL || QSB_ROOT_WARP || QSB_FRONT3_PUBLISH_AB || QSB_SE_BLOCK != 128 || QSB_CTA_WINDOW_SPLIT || QSB_ROOT_LUT_SMEM || QSB_ROOT_PARK_B || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE)
#error "leaf shuffle requires production scalar CTA128 tree with no other arena users"
#endif
#if QSB_TREE_STATIC_N && (ZLAB_TREE != 2 || QSB_SE_BLOCK != 128 || QSB_TREE_UNROLL || QSB_TREE_ROW128 || QSB_ROOT_WARP || QSB_TREE_BOTTOM_FUSE || QSB_TREE_BOTTOM_SHFL || QSB_TREE_LEAF_SHFL)
#error "static tree extent requires the production scalar CTA128 tree"
#endif
#if QSB_TREE_CONST_ADDR && (QSB_TREE_STATIC_N || ZLAB_TREE != 2 || QSB_SE_BLOCK != 128 || QSB_TREE_UNROLL || QSB_TREE_ROW128 || QSB_ROOT_WARP || QSB_TREE_BOTTOM_FUSE || QSB_TREE_BOTTOM_SHFL || QSB_TREE_LEAF_SHFL || QSB_ROOT_LUT_SMEM || QSB_ROOT_PARK_B || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE)
#error "constant tree addresses require the unmodified scalar CTA128 layout"
#endif
#ifndef QSB_ISO_ROOT_SCALE
#define QSB_ISO_ROOT_SCALE 1
#endif
#if QSB_ISO_ROOT_SCALE && !QSB_ISO_FUSED_ROOT_SCALE
__device__ __noinline__ void qsb_iso_scale_tree_inverse(uint64_t *value){
uint64_t invu[5]={QSB_ISO_INVU[0],QSB_ISO_INVU[1],QSB_ISO_INVU[2],QSB_ISO_INVU[3],0};
QSB_TREE_MUL(value,value,invu);
}
#define QSB_ISO_SCALE_ROOT(value) qsb_iso_scale_tree_inverse(value)
#else
#define QSB_ISO_SCALE_ROOT(value) ((void)0)
#endif
#if ZLAB_TREE == 0
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
__shared__ uint64_t tree[4][512];
const int tid=threadIdx.x,n=blockDim.x;
#pragma unroll
for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
__syncthreads();
#pragma unroll 1
for(int width=n>>1;width>0;width>>=1){
if(tid<width){
int node=width+tid;
uint64_t a[5]={0,0,0,0,0},b[5]={0,0,0,0,0};
#pragma unroll
for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
qsb_field_mul(a,a,b);
#pragma unroll
for(int k=0;k<4;k++)tree[k][node]=a[k];
}
if(width>32)__syncthreads();else __syncwarp();
}
if(tid==0){
uint64_t root[5]={0,0,0,0,0};
#pragma unroll
for(int k=0;k<4;k++)root[k]=tree[k][1];
_ModInv(root);
QSB_ISO_SCALE_ROOT(root);
#pragma unroll
for(int k=0;k<4;k++)tree[k][1]=root[k];
}
__syncthreads();
#pragma unroll 1
for(int width=1;width<n;width<<=1){
if(tid<width){
int node=width+tid;
uint64_t parent[5]={0,0,0,0,0},left[5]={0,0,0,0,0},right[5]={0,0,0,0,0};
#pragma unroll
for(int k=0;k<4;k++){
parent[k]=tree[k][node];
left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
}
qsb_field_mul(right,parent,right);qsb_field_mul(left,parent,left);
#pragma unroll
for(int k=0;k<4;k++){
tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
}
}
if((width<<1)>32)__syncthreads();else __syncwarp();
}
#pragma unroll
for(int k=0;k<4;k++)value[k]=tree[k][n+tid];
value[4]=0;
}
#elif ZLAB_TREE == 1
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
__shared__ uint64_t tree[4][512];
const int tid=threadIdx.x,n=blockDim.x;
#pragma unroll
for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
__syncthreads();


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
if(tid==0){


uint64_t a[5],b[5],root[5];
#pragma unroll
for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
a[4]=b[4]=0;
qsb_field_mul_raw(root,a,b);
qsb_field_normalize(root);
_ModInv(root);
root[4]=0;
QSB_ISO_SCALE_ROOT(root);
qsb_field_mul_raw(a,root,a);
qsb_field_mul_raw(b,root,b);
#pragma unroll
for(int k=0;k<4;k++){tree[k][2]=b[k];tree[k][3]=a[k];}
}
__syncwarp();

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


if((width<<1)>32 || (width<<2)==n)__syncthreads();else __syncwarp();
}


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
#else
#ifndef QSB_SC_PARK
#define QSB_SC_PARK 1
#endif
#if QSB_SC_PARK
#if QSB_TREE_ROW128
__shared__ __align__(16) ulonglong2 qsb_sc_products2[2][2 * QSB_SE_BLOCK];
#else
__shared__ uint64_t qsb_sc_products[4][2 * QSB_SE_BLOCK];
#endif
#endif
#if QSB_FRONT3_PUBLISH_AB


__shared__ uint64_t qsb_front_inverse[4][QSB_SE_BLOCK];
#endif
#if QSB_TREE_ROW128
#define QTR_LD4(A,col,v) do{const ulonglong2 qtr0_=(A)[0][(col)],qtr1_=(A)[1][(col)];(v)[0]=qtr0_.x;(v)[1]=qtr0_.y;(v)[2]=qtr1_.x;(v)[3]=qtr1_.y;}while(0)
#define QTR_ST4(A,col,v) do{(A)[0][(col)]=make_ulonglong2((v)[0],(v)[1]);(A)[1][(col)]=make_ulonglong2((v)[2],(v)[3]);}while(0)
#endif
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE



struct QsbTreeNoIdle{__device__ __forceinline__ void operator()()const{}};
#if QSB_ROOT_LUT_SMEM
#define inverses qsb_tree_inverses_smem
#endif



template<int LUT_ISSUED,class Idle,int RW=0>
__device__ __forceinline__ void qsb_block_inverse_tree_x(uint64_t *value,const Idle &idle){
#else
#if QSB_ROOT_PARK_B
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value,uint64_t *parkB=nullptr){
#else
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
#endif
#endif
#if QSB_TREE_ROW128
ulonglong2 (&products2)[2][2 * QSB_SE_BLOCK] = qsb_sc_products2;
__shared__ __align__(16) ulonglong2 inverses2[2][QSB_SE_BLOCK];
#else
#if QSB_SC_PARK
uint64_t (&products)[4][2 * QSB_SE_BLOCK] = qsb_sc_products;
#else
__shared__ uint64_t products[4][2 * QSB_SE_BLOCK];
#endif
#if !QSB_ROOT_LUT_SMEM
#if QSB_FRONT3_PUBLISH_AB
uint64_t (&inverses)[4][QSB_SE_BLOCK] = qsb_front_inverse;
#else
__shared__ uint64_t inverses[4][QSB_SE_BLOCK];
#endif
#endif
#endif
#if QSB_TREE_UNROLL
static_assert(QSB_SE_BLOCK==256,"QSB_TREE_UNROLL (tree.cu): the tree is written out for 256-thread kernel_digest blocks");
const int tid=threadIdx.x;constexpr int n=QSB_SE_BLOCK;
#elif QSB_TREE_STATIC_N
const int tid=threadIdx.x;constexpr int n=QSB_SE_BLOCK;
#else
const int tid=threadIdx.x,n=blockDim.x;
#endif
#if QSB_TREE_CONST_ADDR
constexpr int address_n=QSB_SE_BLOCK;
#else
const int address_n=n;
#endif
#if QSB_TREE_BOTTOM_SHFL || QSB_TREE_LEAF_SHFL
const int leaf_tid=(tid>>1)|((tid&1)*(n>>1));
#else
const int leaf_tid=tid;
#endif
#if QSB_TREE_ROW128
QTR_ST4(products2,tid,value);
#else
#pragma unroll
for(int k=0;k<4;k++)products[k][leaf_tid]=value[k];
#endif
#if QSB_ROOT_LUT_SMEM



if(!LUT_ISSUED)qsb_root_lut_issue(tid,n);
qsb_root_lut_wait();
#endif
__syncthreads();


#if !QSB_TREE_CONST_ADDR
int offset=0;
#endif
#if QSB_TREE_UNROLL
#pragma unroll
#else
#pragma unroll 1
#endif
for(int count=n;count>(QSB_TREE_WAVE_TOP?16:2);count>>=1){
#if QSB_TREE_CONST_ADDR
const int offset=2*address_n-2*count;
#endif
int half=count>>1;
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE
const int ut=(RW && half<=32)?tid-32*RW:tid;
if(RW?(unsigned)ut<(unsigned)half:tid<half){
#else
const int ut=tid;
if(tid<half){
#endif
uint64_t a[5],b[5],out[5];
#if QSB_TREE_ROW128
QTR_LD4(products2,offset+ut,a);QTR_LD4(products2,offset+half+ut,b);
#else
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset+ut];b[k]=products[k][offset+half+ut];}
#endif
a[4]=b[4]=0;
QSB_TREE_MUL(out,a,b);
#if QSB_TREE_ROW128
QTR_ST4(products2,offset+count+ut,out);
#else
#pragma unroll
for(int k=0;k<4;k++)products[k][offset+count+ut]=out[k];
#endif
}
#if !QSB_TREE_CONST_ADDR
offset+=count;
#endif
if(half>32)__syncthreads();else __syncwarp();
}
#if QSB_TREE_CONST_ADDR
constexpr int offset=2*address_n-(QSB_TREE_WAVE_TOP?32:4);
#endif

#if HM43_WARP_ROOT
#ifndef QSB_ROOT_UNIFORM_WARP
#define QSB_ROOT_UNIFORM_WARP 1
#endif
#if QSB_ROOT_UNIFORM_WARP && QSB_INVERSE_LIMBS



if(!QSB_TREE_WAVE_TOP && __all_sync(0xffffffffu,tid<32)){
#else
if(tid<(QSB_INVERSE_LIMBS?32:4)){
#endif
#if !QSB_TREE_ROW128
uint64_t a[5],b[5],root[5];
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
a[4]=b[4]=0;__syncwarp(QSB_INVERSE_LIMBS?0xffffffffu:0x0000000fu);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
root[4]=0;
#if QSB_INVERSE_LIMBS
zi_inverse_limbs(root,tid);
#else
zi_inverse_quad(root,tid);
#endif
if(tid==0)QSB_ISO_SCALE_ROOT(root);
#pragma unroll
for(int k=0;k<4;k++)root[k]=__shfl_sync(QSB_INVERSE_LIMBS?0xffffffffu:0x0000000fu,root[k],0);
if(tid<2){
uint64_t child[5];
#pragma unroll
for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
child[4]=0;QSB_TREE_MUL(child,root,child);
#pragma unroll
for(int k=0;k<4;k++)inverses[k][offset-address_n+tid]=child[k];
}
#endif
}
#if QSB_TREE_WAVE_TOP
#if !(QSB_ROOT_UNIFORM_WARP && QSB_INVERSE_LIMBS)
#error "QSB_TREE_WAVE_TOP is written for the uniform warp-0 root (QSB_ROOT_UNIFORM_WARP, QSB_INVERSE_LIMBS)"
#endif



#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE
if(__all_sync(0xffffffffu,RW?(unsigned)(tid-32*RW)<32u:tid<32)){
const int lt=RW?tid-32*RW:tid;
#else
if(__all_sync(0xffffffffu,tid<32)){
const int lt=tid;
#endif
#if QSB_ROOT_PARK_B
static_assert(QSB_SE_BLOCK>=128,"root B scratch requires three 128-word inverse rows");


if(parkB){
#pragma unroll
for(int j=0;j<12;j++){
volatile uint64_t *p=&inverses[j>>2][32*(j&3)+lt];
*p=parkB[j];
}
}
#endif
const int l8=offset+16,l4=offset+24,l2=offset+28;
const bool cof=lt<16;
uint64_t a[5],b[5],r[5];
#if QSB_TREE_LIVE_MASK
#pragma unroll
for(int k=0;k<5;k++)r[k]=0;
#endif

QSB_TREE_LIVE_IF(lt<8){
#if QSB_TREE_ROW128
QTR_LD4(products2,offset+(lt&7),a);QTR_LD4(products2,offset+8+(lt&7),b);
#else
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset+(lt&7)];b[k]=products[k][offset+8+(lt&7)];}
#endif
a[4]=b[4]=0;QSB_TREE_MUL(r,a,b);
#if !QSB_TREE_TOP_SHFL
if(lt<8){
#if QSB_TREE_ROW128
QTR_ST4(products2,l8+lt,r);
#else
#pragma unroll
for(int k=0;k<4;k++)products[k][l8+lt]=r[k];
#endif
}
#endif
}
#if !QSB_TREE_TOP_SHFL
__syncwarp();
#endif

QSB_TREE_LIVE_IF(lt<20){
#if QSB_TREE_TOP_SHFL

#pragma unroll
for(int k=0;k<4;k++){
const uint64_t pa=__shfl_sync(0xffffffffu,r[k],lt&3);
const uint64_t pb=__shfl_sync(0xffffffffu,r[k],cof?((lt&7)^4):4+(lt&3));
a[k]=cof?products[k][offset+(lt^8)]:pa;b[k]=pb;
}
#else
const int ia=cof?offset+(lt^8):l8+(lt&3);
const int ib=cof?l8+((lt&7)^4):l8+4+(lt&3);
#if QSB_TREE_ROW128
QTR_LD4(products2,ia,a);QTR_LD4(products2,ib,b);
#else
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][ia];b[k]=products[k][ib];}
#endif
#endif
a[4]=b[4]=0;QSB_TREE_MUL(r,a,b);
#if !QSB_TREE_TOP_SHFL
if((unsigned)(lt-16)<4u){
#if QSB_TREE_ROW128
QTR_ST4(products2,l4+(lt&3),r);
#else
#pragma unroll
for(int k=0;k<4;k++)products[k][l4+(lt&3)]=r[k];
#endif
}
#endif
}
#if !QSB_TREE_TOP_SHFL
__syncwarp();
#endif


QSB_TREE_LIVE_IF(lt<18){
#if QSB_TREE_TOP_SHFL

#pragma unroll
for(int k=0;k<4;k++)b[k]=__shfl_sync(0xffffffffu,r[k],16+(cof?((lt&3)^2):2+(lt&1)));
#else
const int ib=cof?l4+((lt&3)^2):l4+2+(lt&1);
#if QSB_TREE_ROW128
QTR_LD4(products2,ib,b);
#else
#pragma unroll
for(int k=0;k<4;k++)b[k]=products[k][ib];
#endif
#endif
b[4]=0;QSB_TREE_MUL(r,r,b);
#if !QSB_TREE_TOP_SHFL
if((unsigned)(lt-16)<2u){
#if QSB_TREE_ROW128
QTR_ST4(products2,l2+(lt&1),r);
#else
#pragma unroll
for(int k=0;k<4;k++)products[k][l2+(lt&1)]=r[k];
#endif
}
#endif
}
#if !QSB_TREE_TOP_SHFL
__syncwarp();
#endif

QSB_TREE_LIVE_IF(lt<17){
#if QSB_TREE_TOP_SHFL

#pragma unroll
for(int k=0;k<4;k++)b[k]=__shfl_sync(0xffffffffu,r[k],16+(cof?((lt&1)^1):1));
#else
const int ib=cof?l2+((lt&1)^1):l2+1;
#if QSB_TREE_ROW128
QTR_LD4(products2,ib,b);
#else
#pragma unroll
for(int k=0;k<4;k++)b[k]=products[k][ib];
#endif
#endif
b[4]=0;QSB_TREE_MUL(r,r,b);
}
uint64_t root[5];
#pragma unroll
for(int k=0;k<4;k++)root[k]=__shfl_sync(0xffffffffu,r[k],16);
qsb_field_normalize(root);
root[4]=0;
zi_inverse_limbs(root,lt);
if(lt==0)QSB_ISO_SCALE_ROOT(root);
#pragma unroll
for(int k=0;k<4;k++)root[k]=__shfl_sync(0xffffffffu,root[k],0);
#if QSB_ROOT_PARK_B

if(parkB){
#pragma unroll
for(int j=0;j<12;j++){
volatile const uint64_t *p=&inverses[j>>2][32*(j&3)+lt];
parkB[j]=*p;
}
__syncwarp();
}
#endif

QSB_TREE_LIVE_IF(cof){r[4]=0;QSB_TREE_MUL(r,root,r);}
if(cof){
#if QSB_TREE_ROW128
QTR_ST4(inverses2,offset-n+lt,r);
#else
#pragma unroll
for(int k=0;k<4;k++)inverses[k][offset-address_n+lt]=r[k];
#endif
}
}
#if QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE
else idle();
#endif
#endif
#elif HM41_QUAD_ROOT
if(tid<4){
uint64_t a[5],b[5],root[5];
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
a[4]=b[4]=0;__syncwarp(0x0000000f);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
root[4]=0;hm41_quad_inverse(root,tid);
if(tid==0)QSB_ISO_SCALE_ROOT(root);
#pragma unroll
for(int k=0;k<4;k++)root[k]=__shfl_sync(0x0000000f,root[k],0);
if(tid<2){
uint64_t child[5];
#pragma unroll
for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
child[4]=0;QSB_TREE_MUL(child,root,child);
#pragma unroll
for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
}
}
#elif HM39_PAIR_ROOT
if(tid<2){
uint64_t a[5],b[5],root[5];
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
a[4]=b[4]=0;__syncwarp(0x00000003);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
root[4]=0;hm39_pair_inverse(root,tid);
if(tid==0)QSB_ISO_SCALE_ROOT(root);
#pragma unroll
for(int k=0;k<4;k++)root[k]=__shfl_sync(0x00000003,root[k],0);
uint64_t child[5];
#pragma unroll
for(int k=0;k<4;k++)child[k]=tid? a[k]:b[k];
child[4]=0;QSB_TREE_MUL(child,root,child);
#pragma unroll
for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
}
#else
if(tid==0){
uint64_t a[5],b[5],root[5];
#pragma unroll
for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
a[4]=b[4]=0;
QSB_TREE_MUL(root,a,b);
qsb_field_normalize(root);
_ModInv(root);
root[4]=0;
QSB_ISO_SCALE_ROOT(root);
QSB_TREE_MUL(a,root,a);
QSB_TREE_MUL(b,root,b);

#pragma unroll
for(int k=0;k<4;k++){inverses[k][offset-n]=b[k];inverses[k][offset-n+1]=a[k];}
}
#endif
__syncwarp();


#if !QSB_TREE_CONST_ADDR
offset-=QSB_TREE_WAVE_TOP?32:4;
#endif
#if QSB_TREE_UNROLL
#pragma unroll
#else
#pragma unroll 1
#endif
for(int count=QSB_TREE_WAVE_TOP?32:4;count<((QSB_TREE_BOTTOM_FUSE||QSB_TREE_BOTTOM_SHFL||QSB_TREE_LEAF_SHFL)?(n>>1):n);count<<=1){
#if QSB_TREE_CONST_ADDR
const int offset=2*address_n-2*count;
#endif
int half=count>>1;
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE
const int ut=(RW && count<=32)?tid-32*RW:tid;
if(RW?(unsigned)ut<(unsigned)count:tid<count){
#else
const int ut=tid;
if(tid<count){
#endif
uint64_t parent_inv[5],sibling[5],child_inv[5];
#if QSB_TREE_ROW128
QTR_LD4(inverses2,offset+count-n+(ut&(half-1)),parent_inv);QTR_LD4(products2,offset+(ut^half),sibling);
#else
#pragma unroll
for(int k=0;k<4;k++){
parent_inv[k]=inverses[k][offset+count-address_n+(ut&(half-1))];
sibling[k]=products[k][offset+(ut^half)];
}
#endif
parent_inv[4]=sibling[4]=0;
QSB_TREE_MUL(child_inv,parent_inv,sibling);
#if QSB_TREE_ROW128
QTR_ST4(inverses2,offset-n+ut,child_inv);
#else
#pragma unroll
for(int k=0;k<4;k++)inverses[k][offset-address_n+ut]=child_inv[k];
#endif
}
#if !QSB_TREE_CONST_ADDR
offset-=count<<1;
#endif
if((count<<1)>32)__syncthreads();else __syncwarp();
}

{
const int half=address_n>>1;
uint64_t parent_inv[5],sibling[5];
#if QSB_TREE_BOTTOM_SHFL || QSB_TREE_LEAF_SHFL

#pragma unroll
for(int k=0;k<5;k++)parent_inv[k]=0;
if(!(tid&1)){
const int j=tid>>1;
#pragma unroll
for(int k=0;k<4;k++){
parent_inv[k]=inverses[k][half+(j&((half>>1)-1))];
sibling[k]=products[k][n+(j^(half>>1))];
}
sibling[4]=0;
QSB_TREE_MUL(parent_inv,parent_inv,sibling);
}

#pragma unroll
for(int k=0;k<4;k++)parent_inv[k]=__shfl_sync(0xffffffffu,parent_inv[k],(tid&31)&~1);
#pragma unroll
for(int k=0;k<4;k++)sibling[k]=products[k][leaf_tid^half];
#elif QSB_TREE_BOTTOM_FUSE



#pragma unroll
for(int k=0;k<4;k++){
parent_inv[k]=inverses[k][half+(tid&((half>>1)-1))];
sibling[k]=products[k][n+((tid&(half-1))^(half>>1))];
}
parent_inv[4]=sibling[4]=0;
QSB_TREE_MUL(parent_inv,parent_inv,sibling);
#pragma unroll
for(int k=0;k<4;k++)sibling[k]=products[k][tid^half];
#else
#if QSB_TREE_ROW128
QTR_LD4(inverses2,tid&(half-1),parent_inv);QTR_LD4(products2,tid^half,sibling);
#else
#pragma unroll
for(int k=0;k<4;k++){
parent_inv[k]=inverses[k][tid&(half-1)];
sibling[k]=products[k][tid^half];
}
#endif
#endif
parent_inv[4]=sibling[4]=0;
QSB_TREE_MUL(value,parent_inv,sibling);
}
value[4]=0;
}
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_DEN_CROSS_IDLE
#if QSB_ROOT_LUT_SMEM
#undef inverses
#endif
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
qsb_block_inverse_tree_x<0>(value,QsbTreeNoIdle());
}
#endif
#endif
#if QSB_DEN_CROSS_IDLE && ZLAB_TREE != 2
#error "QSB_DEN_CROSS_IDLE requires the level-packed tree"
#endif
#if (QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT) && ZLAB_TREE != 2
#error "QSB_ROOT_LUT_SMEM and QSB_PRE3_ROOT (tree.cu) are written for the level-packed tree (ZLAB_TREE 2)"
#endif
#if QSB_ROOT_LUT_SMEM && !(HM43_WARP_ROOT && QSB_INVERSE_LIMBS)
#error "QSB_ROOT_LUT_SMEM (tree.cu) is written for the warp-0 limbs root (HM43_WARP_ROOT, QSB_INVERSE_LIMBS)"
#endif
#if QSB_TREE_WAVE_TOP && (ZLAB_TREE != 2 || !HM43_WARP_ROOT)
#error "QSB_TREE_WAVE_TOP (tree.cu) is written for the level-packed tree (ZLAB_TREE 2) with the warp-0 root (HM43_WARP_ROOT)"
#endif

#if QSB_TREE_ROW128 && (ZLAB_TREE != 2 || !QSB_TREE_WAVE_TOP || !QSB_SC_PARK || \
QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_PARK_B || QSB_TREE_TOP_SHFL || \
!HM43_WARP_ROOT || !QSB_INVERSE_LIMBS)
#error "QSB_TREE_ROW128 requires separate shared paired wave-top arenas and uniform limbs root"
#endif
