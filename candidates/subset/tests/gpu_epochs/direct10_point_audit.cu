/* Small-row audit for the direct ten-term XYZZ chain. */
#define QSB_DIRECT10 1
#define QSB_D10_COMPACT_AUDIT 1
#define main qsb_candidate_main
#include "tree.cu"
#undef main
#include <vector>

struct qsb_d10_audit_case {
    uint64_t k[4];
    uint64_t base[4];
    uint64_t rows[10][8];
};
struct qsb_d10_audit_result {
    uint64_t words[16];
    uint32_t bad;
    uint32_t pad;
};
static_assert(sizeof(qsb_d10_audit_case)==704,"compact table fixture layout");

__global__ void qsb_d10_audit_chain(const qsb_d10_audit_case *cases,
                                    qsb_d10_audit_result *out, unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4];
    uint32_t bad=0;
    qsb_filter_chain_trial(X,Y,ZZ,ZZZ,cases[i].k,(const uint8_t*)cases[i].rows,bad);
    #pragma unroll
    for(int j=0;j<4;j++) {
        out[i].words[j]=X[j]; out[i].words[4+j]=Y[j];
        out[i].words[8+j]=ZZ[j]; out[i].words[12+j]=ZZZ[j];
    }
    out[i].bad=bad; out[i].pad=0;
}

static uint32_t audit_field(const uint64_t k[4],unsigned start,unsigned width) {
    uint32_t f=0;
    for(unsigned b=0;b<width;b++)
        f|=(uint32_t)((k[(start+b)>>6]>>((start+b)&63u))&1u)<<b;
    return f;
}
static void audit_set_field(uint64_t k[4],unsigned start,unsigned width,uint32_t f) {
    for(unsigned b=0;b<width;b++) {
        const unsigned pos=start+b;
        const uint64_t mask=1ULL<<(pos&63u);
        if((f>>b)&1u) k[pos>>6]|=mask; else k[pos>>6]&=~mask;
    }
}
static uint64_t audit_rng(uint64_t &s) {
    s^=s<<13; s^=s>>7; s^=s<<17; return s;
}
static void audit_store_le(const BIGNUM *v,uint64_t out[4]) {
    if(BN_bn2lebinpad(v,(uint8_t*)out,32)!=32) abort();
}
static bool audit_make_case(qsb_d10_audit_case &dst,const uint64_t k[4],
                            const BIGNUM *base_scalar,EC_GROUP *grp,BN_CTX *ctx,
                            const BIGNUM *order,const BIGNUM *p,
                            BIGNUM *coeff,BIGNUM *scalar,BIGNUM *x,BIGNUM *y,
                            BIGNUM *alpha,BIGNUM *beta,EC_POINT *pt) {
    static const unsigned widths[10]={25,26,26,26,26,26,26,25,25,25};
    static const unsigned shifts[10]={0,25,51,77,103,129,155,181,206,231};
    memcpy(dst.k,k,32);
    audit_store_le(base_scalar,dst.base);
    if(qsb_d10_incomplete_exception(k)) return true;
    for(int c=0;c<10;c++) {
        BN_zero(coeff);
        if(c==0) {
            BN_one(coeff); BN_lshift(coeff,coeff,255); BN_sub_word(coeff,1u<<24);
            BN_add_word(coeff,audit_field(k,shifts[c],widths[c]));
        } else {
            const uint32_t f=audit_field(k,shifts[c],widths[c]);
            const int32_t d=(int32_t)(2u*f+1u-(1u<<widths[c]));
            BN_set_word(coeff,(BN_ULONG)(d<0?-d:d));
            BN_lshift(coeff,coeff,(int)shifts[c]-1);
        }
        if(!BN_mod_mul(scalar,coeff,(BIGNUM*)base_scalar,order,ctx)) return false;
        if(!EC_POINT_mul(grp,pt,scalar,NULL,NULL,ctx)) return false;
        gt_point_to_limbs(grp,pt,x,y,alpha,beta,p,ctx,dst.rows[c]);
#if QSB_SMEM02
        dst.rows[c][4] += QSB_SMEM02_Q;
#endif
    }
    return true;
}

int main() {
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *order=BN_new(),*p=BN_new(),*base=BN_new(),*coeff=BN_new(),*scalar=BN_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*alpha=BN_new(),*beta=BN_new(),*inv=BN_new(),*got=BN_new();
    BIGNUM *expected_scalar=BN_new(),*expected_x=BN_new(),*expected_y=BN_new();
    EC_POINT *pt=EC_POINT_new(grp),*want=EC_POINT_new(grp);
    if(!grp||!ctx||!order||!p||!base||!coeff||!scalar||!x||!y||!alpha||!beta||!inv||!got||
       !expected_scalar||!expected_x||!expected_y||!pt||!want) return 2;
    EC_GROUP_get_order(grp,order,ctx); EC_GROUP_get_curve_GFp(grp,p,NULL,NULL,ctx);
    BN_one(alpha); BN_one(beta);
    std::vector<qsb_d10_audit_case> cases;
    cases.reserve(80);
    uint64_t seed=0xD1047A11C0FFEE01ULL;
    static const unsigned widths[10]={25,26,26,26,26,26,26,25,25,25};
    static const unsigned shifts[10]={0,25,51,77,103,129,155,181,206,231};
    for(unsigned base_id=0;base_id<3;base_id++) {
        uint8_t seed_bytes[32],base_bytes[32];
        for(unsigned j=0;j<32;j++) seed_bytes[j]=(uint8_t)(j*47u+base_id*101u+19u);
        SHA256(seed_bytes,sizeof seed_bytes,base_bytes);
        BN_bin2bn(base_bytes,32,base); BN_mod(base,base,order,ctx);
        if(BN_is_zero(base)) BN_one(base);
        for(int c=1;c<10;c++) {
            for(unsigned side=0;side<2;side++) {
                uint64_t k[4]={audit_rng(seed),audit_rng(seed),audit_rng(seed),audit_rng(seed)};
                const uint32_t half=1u<<(widths[c]-1u);
                audit_set_field(k,shifts[c],widths[c],side?half:half-1u);
                qsb_d10_audit_case item{};
                if(!audit_make_case(item,k,base,grp,ctx,order,p,coeff,scalar,x,y,alpha,beta,pt)) return 2;
                cases.push_back(item);
            }
        }
        for(unsigned j=0;j<4;j++) {
            uint64_t k[4]={audit_rng(seed),audit_rng(seed),audit_rng(seed),audit_rng(seed)};
            qsb_d10_audit_case item{};
            if(!audit_make_case(item,k,base,grp,ctx,order,p,coeff,scalar,x,y,alpha,beta,pt)) return 2;
            cases.push_back(item);
        }
    }
    const uint64_t exceptional[4][4]={
        {0,0,0,0},
        {0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL},
        {0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFF8000000000ULL},
        {0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0x0000007FFFFFFFFFULL}
    };
    BN_one(base);
    for(const auto &k:exceptional) {
        qsb_d10_audit_case item{};
        if(!audit_make_case(item,k,base,grp,ctx,order,p,coeff,scalar,x,y,alpha,beta,pt)) return 2;
        cases.push_back(item);
    }
    qsb_d10_audit_case *d_cases=nullptr; qsb_d10_audit_result *d_out=nullptr;
    if(cudaMalloc(&d_cases,cases.size()*sizeof(cases[0]))!=cudaSuccess ||
       cudaMalloc(&d_out,cases.size()*sizeof(qsb_d10_audit_result))!=cudaSuccess) return 2;
    if(cudaMemcpy(d_cases,cases.data(),cases.size()*sizeof(cases[0]),cudaMemcpyHostToDevice)!=cudaSuccess) return 2;
    qsb_d10_audit_chain<<<(cases.size()+63)/64,64>>>(d_cases,d_out,(unsigned)cases.size());
    if(cudaDeviceSynchronize()!=cudaSuccess) return 2;
    std::vector<qsb_d10_audit_result> results(cases.size());
    if(cudaMemcpy(results.data(),d_out,results.size()*sizeof(results[0]),cudaMemcpyDeviceToHost)!=cudaSuccess) return 2;
    unsigned bad=0,checked=0,dropped=0;
    for(unsigned i=0;i<cases.size();i++) {
        const qsb_d10_audit_case &item=cases[i];
        const qsb_d10_audit_result &r=results[i];
        if(qsb_d10_incomplete_exception(item.k)) {
            bool all_zero=true; for(uint64_t w:r.words) all_zero &= w==0;
            if(!all_zero || r.bad==0) { if(bad++<8) printf("exceptional scalar was not dropped at case %u\n",i); }
            else dropped++;
            continue;
        }
        BN_lebin2bn((const uint8_t*)item.k,32,coeff);
        BN_lebin2bn((const uint8_t*)item.base,32,base);
        BN_mod_mul(expected_scalar,coeff,base,order,ctx);
        if(!EC_POINT_mul(grp,want,expected_scalar,NULL,NULL,ctx) || EC_POINT_is_at_infinity(grp,want)) {
            if(bad++<8) printf("unexpected infinity at case %u\n",i); continue;
        }
        BN_lebin2bn((const uint8_t*)r.words,32,got); BN_nnmod(got,got,p,ctx);
        BN_lebin2bn((const uint8_t*)(r.words+4),32,y); BN_nnmod(y,y,p,ctx);
        BN_lebin2bn((const uint8_t*)(r.words+8),32,x); BN_nnmod(x,x,p,ctx);
        BN_lebin2bn((const uint8_t*)(r.words+12),32,inv); BN_nnmod(inv,inv,p,ctx);
        if(BN_is_zero(x)||BN_is_zero(inv)||BN_mod_inverse(x,x,p,ctx)==NULL||BN_mod_inverse(inv,inv,p,ctx)==NULL) {
            if(bad++<8) printf("invalid XYZZ denominator at case %u\n",i); continue;
        }
        BN_mod_mul(got,got,x,p,ctx); BN_mod_mul(y,y,inv,p,ctx);
#if QSB_YNEG_FOLD
        BN_mod_sub(y,p,y,p,ctx);
#endif
        EC_POINT_get_affine_coordinates_GFp(grp,want,expected_x,expected_y,ctx);
        if(BN_cmp(got,expected_x)!=0||BN_cmp(y,expected_y)!=0) {
            if(bad++<8) printf("point mismatch base=%u case=%u badflag=%u\n",i/22,i,r.bad);
        } else checked++;
    }
    printf("Direct ten-term point audit: %u points matched, %u exceptional inputs dropped, %u errors\n",
           checked,dropped,bad);
    cudaFree(d_cases); cudaFree(d_out);
    EC_POINT_free(pt); EC_POINT_free(want); EC_GROUP_free(grp); BN_CTX_free(ctx);
    BN_free(order);BN_free(p);BN_free(base);BN_free(coeff);BN_free(scalar);BN_free(x);BN_free(y);
    BN_free(alpha);BN_free(beta);BN_free(inv);BN_free(got);BN_free(expected_scalar);BN_free(expected_x);BN_free(expected_y);
    return bad?1:0;
}
