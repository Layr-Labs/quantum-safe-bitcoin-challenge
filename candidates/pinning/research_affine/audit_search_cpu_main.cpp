// Appended to actual extracted helpers/search body by audit_search_cpu.py.
static bool encode_cpu(QsbShiftedTailPoint& out,const EC_GROUP*g,const EC_POINT*p,BN_CTX*ctx) {
    BIGNUM*x=BN_new(),*y=BN_new();
    const bool ok=EC_POINT_get_affine_coordinates(g,p,x,y,ctx)==1&&
      BN_bn2lebinpad(x,(uint8_t*)out.x,32)==32&&BN_bn2lebinpad(y,(uint8_t*)out.y,32)==32;
    BN_free(x);BN_free(y);return ok;
}
int main(int argc,char**argv) {
    assert(argc==3);
    constexpr unsigned workers=2,batch=513;
    constexpr uint32_t sequence=0x80000013u,start_lt=0xfffffdffu;
    pinning2_params_t pp{};assert(load_pinning2(argv[1],&pp)==0);
    std::ifstream file(argv[2],std::ios::binary);
    std::vector<uint8_t> prefix((std::istreambuf_iterator<char>(file)),{});
    assert(prefix.size()==9920&&pp.suffix_len==75&&pp.total_preimage_len==9995);
    std::vector<uint8_t> pre=prefix;pre.insert(pre.end(),pp.suffix,pp.suffix+pp.suffix_len);
    for(unsigned b=0;b<4;b++)pre[prefix.size()+pp.seq_offset+b]=uint8_t(sequence>>(8*b));
    SHA256_CTX context;SHA256_Init(&context);
    SHA256_Update(&context,pre.data(),prefix.size()+64);
    pin_tail_words[0]=(uint32_t(pp.suffix[64])<<24)|(uint32_t(pp.suffix[65])<<16)|(uint32_t(pp.suffix[66])<<8);
    pin_tail_words[1]=pp.suffix[71];
    pin_tail_words[2]=(uint32_t(pp.suffix[72])<<24)|(uint32_t(pp.suffix[73])<<16)|(uint32_t(pp.suffix[74])<<8)|0x80;
    qsb_tail_pre tp;qsb_make_tail_pre(&tp,context.h,pin_tail_words[2]);
    EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);assert(g);BN_CTX*ctx=BN_CTX_new();
    BIGNUM*n=BN_new(),*nri=BN_lebin2bn(pp.neg_r_inv,32,nullptr),*scalar=BN_new(),*inv2=BN_new(),*part=BN_new();
    BIGNUM*x=BN_lebin2bn(pp.u2r_x,32,nullptr),*y=BN_lebin2bn(pp.u2r_y,32,nullptr);
    EC_POINT*point=EC_POINT_new(g),*recovery=EC_POINT_new(g),*minus_recovery=EC_POINT_new(g),*sum=EC_POINT_new(g);
    assert(EC_GROUP_get_order(g,n,ctx)==1&&BN_one(inv2)==1&&BN_lshift1(inv2,inv2)==1);
    assert(BN_mod_inverse(inv2,inv2,n,ctx));
    assert(EC_POINT_set_affine_coordinates(g,recovery,x,y,ctx)==1);
    assert(EC_POINT_copy(minus_recovery,recovery)==1&&EC_POINT_invert(g,minus_recovery,ctx)==1);
    std::vector<uint8_t> table(64ull*GT_TOTAL_ENTRIES,0);
    std::vector<QsbShiftedTailRecord> tails(65536);
    std::vector<bool> populated(GT_TOTAL_ENTRIES,false),tail_populated(65536,false);
    std::array<std::array<uint8_t,32>,batch> hashes{};
    unsigned ordinary_entries=0,tail_entries=0;
    for(unsigned c=0;c<batch;c++) {
        const uint32_t lt=start_lt+c;
        for(unsigned b=0;b<4;b++)pre[prefix.size()+pp.lt_offset+b]=uint8_t(lt>>(8*b));
        uint8_t first[32];SHA256(pre.data(),pre.size(),first);SHA256(first,32,hashes[c].data());
        uint64_t z[4];qsb_affine_search_hash_scalar(z,lt,tp);
        for(unsigned b=0;b<32;b++)assert(((uint8_t*)z)[b]==hashes[c][31-b]);
        uint32_t codes[15][128];threadIdx.x=0;qsb_affine_search_recode(z,codes);
        for(unsigned w=0;w<15;w++) {
            const unsigned index=codes[w][0]&0x1ffffu;
            if((w<14&&populated[gt_offset(w)+index])||(w==14&&tail_populated[index]))continue;
            assert(BN_set_word(part,2*index+1)==1&&BN_lshift(part,part,gt_shift(w))==1);
            assert(BN_mod_mul(scalar,part,inv2,n,ctx)==1&&BN_mod_mul(scalar,scalar,nri,n,ctx)==1);
            assert(EC_POINT_mul(g,point,scalar,nullptr,nullptr,ctx)==1);
            if(w<14) {
                QsbShiftedTailPoint encoded;assert(encode_cpu(encoded,g,point,ctx));
                memcpy(table.data()+64ull*(gt_offset(w)+index),&encoded,64);
                populated[gt_offset(w)+index]=true;++ordinary_entries;
            } else {
                for(unsigned arm=0;arm<2;arm++) {
                    assert(EC_POINT_add(g,sum,point,arm?minus_recovery:recovery,ctx)==1);
                    assert(encode_cpu(tails[index].arm[arm],g,sum,ctx));
                }
                tail_populated[index]=true;++tail_entries;
            }
        }
    }
    std::array<qsb_inverse_service::Slot,workers> slots{};
    std::array<CpuBlock,workers> blocks;
    constexpr uint32_t canary=0x7dbba5d3;
    std::vector<uint32_t> hits((batch+15)/16+1,0),exceptions((batch+31)/32+1,0);
    hits.back()=exceptions.back()=canary;
    std::array<QsbAffineAudit,batch> audit{};
    std::vector<std::thread> threads;
    for(unsigned lane=0;lane<32;lane++)threads.emplace_back([&,lane] {
        threadIdx.x=lane;blockIdx.x=0;cpu_block=nullptr;
        qsb_affine_search<true>(table.data(),tails.data(),slots.data(),hits.data(),exceptions.data(),batch,start_lt,tp,1,workers,audit.data());
    });
    for(unsigned c=0;c<workers*128;c++)threads.emplace_back([&,c] {
        unsigned worker=c/128;threadIdx.x=c%128;blockIdx.x=worker+1;cpu_block=&blocks[worker];
        qsb_affine_search<true>(table.data(),tails.data(),slots.data(),hits.data(),exceptions.data(),batch,start_lt,tp,1,workers,audit.data());
    });
    for(auto &thread:threads)thread.join();
    assert(hits.back()==canary&&exceptions.back()==canary);
    assert((hits[(batch-1)/16]&~3u)==0); // only idx512 occupies the last mask word
    assert(exceptions[(batch-1)/32]==0);
    unsigned mismatch=0,declined=0,match_pairs=0,matching_keys=0,both_hits=0;
    for(unsigned c=0;c<batch;c++) {
        if(!audit[c].usable||((exceptions[c/32]>>(c%32))&1u)){++declined;continue;}
        assert(BN_bin2bn(hashes[c].data(),32,part)&&BN_mod_mul(scalar,part,nri,n,ctx)==1);
        assert(EC_POINT_mul(g,point,scalar,nullptr,nullptr,ctx)==1);
        uint32_t expected_mask=0;
        for(unsigned arm=0;arm<2;arm++) {
            assert(EC_POINT_add(g,sum,point,arm?minus_recovery:recovery,ctx)==1);
            QsbShiftedTailPoint expected;assert(encode_cpu(expected,g,sum,ctx));
            if(memcmp(expected.x,audit[c].x[arm],32)||((expected.y[0]&1u)!=((audit[c].parities>>arm)&1u)))++mismatch;
            uint8_t compressed[33],digest[32];
            assert(EC_POINT_point2oct(g,sum,POINT_CONVERSION_COMPRESSED,compressed,33,ctx)==33);
            SHA256(compressed,33,digest);
            if((digest[0]>>4)==0){expected_mask|=1u<<arm;++matching_keys;}
        }
        if(((hits[c/16]>>(2*(c%16)))&3u)!=expected_mask)++mismatch;
        if(expected_mask)++match_pairs;if(expected_mask==3)++both_hits;
    }
    assert(slots[0].request==qsb_inverse_service::STOP&&slots[0].response==42);
    assert(slots[1].request==qsb_inverse_service::STOP&&slots[1].response==28);
    printf("{\"active_candidates\":%u,\"inactive_lane_iterations\":127,\"worker_ctas\":2,\"service_lanes\":32,\"root_roundtrips\":70,\"checked_affine_arms\":%u,\"ordinary_table_entries\":%u,\"shifted_table_entries\":%u,\"matching_candidates_N4\":%u,\"matching_recids_N4\":%u,\"both_hit_candidates_N4\":%u,\"mismatches\":%u,\"declined\":%u,\"actual_search_kernel\":true,\"gpu_execution\":false}\n",
           batch,batch*2,ordinary_entries,tail_entries,match_pairs,matching_keys,both_hits,mismatch,declined);
    EC_POINT_free(point);EC_POINT_free(recovery);EC_POINT_free(minus_recovery);EC_POINT_free(sum);
    BN_free(n);BN_free(nri);BN_free(scalar);BN_free(inv2);BN_free(part);BN_free(x);BN_free(y);EC_GROUP_free(g);BN_CTX_free(ctx);free(pp.suffix);
    return mismatch||declined?1:0;
}
