// SPDX-License-Identifier: GPL-3.0-only
// Collective affine fixed-base accumulation. Uses the inherited signed table
// and inverse tree; the independent exact hit replay remains unchanged.
#pragma once

__device__ __forceinline__ void qsb_affine_chain(
    uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
    const uint64_t k[4], const uint8_t *gTable) {
    uint64_t M[4]; int sign;
    gt_recode_setup(k,M,&sign);
    const uint64_t sflag=(uint64_t)(sign<0);
    uint32_t idx; uint64_t neg;
    gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
    gt_load_signed(gTable,0,idx,neg,X,Y);
    bool infinity=false;
    #pragma unroll 1
    for(int c=1;c<GT_CHUNKS;c++) {
        uint64_t bx[4],by[4],den[5];
        gt_direct_digit(M,sflag,(unsigned)gt_shift(c)+1u,gt_width(c),
                        c==GT_CHUNKS-1,&idx,&neg);
        gt_load_signed(gTable,c,idx,neg,bx,by);
        _ModSub256(den,bx,X);
        qsb_field_normalize(den);
        const bool same_x=(den[0]|den[1]|den[2]|den[3])==0;
        const bool same_y=((by[0]^Y[0])|(by[1]^Y[1])|(by[2]^Y[2])|(by[3]^Y[3]))==0;
        // Keep only one mode word live across the root inverse; reconstruct
        // the numerator afterward instead of parking another eight registers.
        // 0: ordinary addition, 1: infinity input, 2: infinity output, 3: double.
        int mode=infinity?1:(same_x?(same_y?3:2):0);
        if(mode==3) {
            _ModAdd256(den,Y,Y);
            qsb_field_normalize(den);
        }
        const bool zero_den=(den[0]|den[1]|den[2]|den[3])==0;
        if(mode!=1 && zero_den)mode=2;
        // Every lane participates, including infinity, cancellation and tails.
        // Identity factors keep one exceptional lane from poisoning its block.
        if(mode==1 || mode==2) {
            den[0]=1;den[1]=den[2]=den[3]=0;
        }
        den[4]=0;
        qsb_block_inverse_tree(den);
        // The tree's final sibling reads have no trailing block barrier.
        // Its shared storage is reused at the next addition AND by the paired
        // finish, so all lanes must finish reading before any can re-enter it.
        __syncthreads();
        if(mode==1) {
            Load256(X,bx);Load256(Y,by);infinity=false;
        } else if(mode==2) {
            X[0]=X[1]=X[2]=X[3]=0;
            Y[0]=Y[1]=Y[2]=Y[3]=0;
            infinity=true;
        } else {
            uint64_t num[4],slope[4],nx[4],ny[4];
            if(mode==3){
                _ModSqr(num,X);
                _ModAdd256(slope,num,num);
                _ModAdd256(num,num,slope);
            }else{
                _ModSub256(num,by,Y);
            }
            _ModMult(slope,num,den);
            _ModSqr(nx,slope);
            _ModSub256(nx,nx,X);
            _ModSub256(nx,nx,bx);
            _ModSub256(ny,X,nx);
            _ModMult(ny,ny,slope);
            _ModSub256(ny,ny,Y);
            qsb_field_normalize(nx);qsb_field_normalize(ny);
            Load256(X,nx);Load256(Y,ny);
        }
    }
    ZZ[0]=ZZZ[0]=infinity?0:1;
    ZZ[1]=ZZ[2]=ZZ[3]=ZZZ[1]=ZZZ[2]=ZZZ[3]=0;
}
