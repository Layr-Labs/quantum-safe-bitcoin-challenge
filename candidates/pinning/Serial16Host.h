// Exact per-problem constants for the two serial16 final-doubling scalars.
#pragma once
static bool qsb_make_serial16_special(uint64_t out[12],const uint8_t nri_bytes[32]) {
    int ok=0;uint8_t bytes[96];
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    if(!grp)return false;
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *nri=BN_new(),*k=BN_new(),*order=BN_new(),*p=BN_new(),*x=BN_new(),*y=BN_new(),*ny=BN_new();
    EC_POINT *point=EC_POINT_new(grp);
    if(!ctx||!nri||!k||!order||!p||!x||!y||!ny||!point)goto done;
    if(!BN_lebin2bn(nri_bytes,32,nri)||!BN_set_word(k,65535)||!BN_lshift(k,k,240))goto done;
    if(!EC_GROUP_get_order(grp,order,ctx)||!BN_mod_mul(k,k,nri,order,ctx))goto done;
    if(!EC_POINT_mul(grp,point,k,NULL,NULL,ctx)||EC_POINT_is_at_infinity(grp,point))goto done;
    if(!EC_POINT_get_affine_coordinates_GFp(grp,point,x,y,ctx))goto done;
    if(!EC_GROUP_get_curve_GFp(grp,p,NULL,NULL,ctx)||!BN_sub(ny,p,y))goto done;
    if(BN_bn2lebinpad(x,bytes,32)!=32||BN_bn2lebinpad(y,bytes+32,32)!=32||BN_bn2lebinpad(ny,bytes+64,32)!=32)goto done;
    memcpy(out,bytes,sizeof(bytes));ok=1;
done:
    EC_POINT_free(point);BN_free(nri);BN_free(k);BN_free(order);BN_free(p);BN_free(x);BN_free(y);BN_free(ny);
    BN_CTX_free(ctx);EC_GROUP_free(grp);return ok!=0;
}
