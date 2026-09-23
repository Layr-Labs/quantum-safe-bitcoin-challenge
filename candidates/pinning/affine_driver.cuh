// SPDX-License-Identifier: GPL-3.0-only
// Runtime integration of the experimental affine worker/inversion service.
#pragma once
#include <vector>
#include <errno.h>

#define QSB_AFFINE_CUDA(call) do {cudaError_t e=(call);if(e!=cudaSuccess){ \
    fprintf(stderr,"Affine CUDA failure (%s): %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)

struct QsbAffineGate {
    EC_GROUP* group=nullptr;BN_CTX* ctx=nullptr;
    BIGNUM* order=nullptr;BIGNUM* nri=nullptr;EC_POINT* recovery=nullptr;
    bool init(const pinning2_params_t& pp) {
        group=EC_GROUP_new_by_curve_name(NID_secp256k1);ctx=BN_CTX_new();
        order=BN_new();nri=BN_lebin2bn(pp.neg_r_inv,32,nullptr);
        if(!group||!ctx||!order||!nri)return false;
        recovery=EC_POINT_new(group);
        BIGNUM* x=BN_lebin2bn(pp.u2r_x,32,nullptr);BIGNUM* y=BN_lebin2bn(pp.u2r_y,32,nullptr);
        bool ok=recovery&&x&&y&&EC_GROUP_get_order(group,order,ctx)==1&&
            EC_POINT_set_affine_coordinates(group,recovery,x,y,ctx)==1;
        BN_free(x);BN_free(y);return ok;
    }
    ~QsbAffineGate(){EC_POINT_free(recovery);BN_free(order);BN_free(nri);BN_CTX_free(ctx);EC_GROUP_free(group);}
};

static qsb_tail_pre qsb_affine_sequence_params(const pinning2_params_t& pp,uint32_t sequence,uint32_t tail_w2) {
    uint8_t block[64];memcpy(block,pp.suffix,64);
    for(unsigned i=0;i<4;i++)block[pp.seq_offset+i]=uint8_t(sequence>>(8*i));
    SHA256_CTX ctx;SHA256_Init(&ctx);
    for(unsigned i=0;i<8;i++)ctx.h[i]=pp.midstate[i];
    SHA256_Transform(&ctx,block);
    uint32_t mid[8];for(unsigned i=0;i<8;i++)mid[i]=ctx.h[i];
    qsb_tail_pre tp;qsb_make_tail_pre(&tp,mid,tail_w2);return tp;
}

// Independent OpenSSL reconstruction for the startup device audit. It never
// reads the generated fixed-base tables or the GPU's SHA/recode intermediates.
static bool qsb_affine_reference(const pinning2_params_t& pp,uint32_t sequence,uint32_t lt,
    QsbAffineGate& gate,QsbAffineAudit& expected,uint32_t& hit_bits) {
    uint8_t block[128]={0};memcpy(block,pp.suffix,pp.suffix_len);
    for(unsigned i=0;i<4;i++){
        block[pp.seq_offset+i]=uint8_t(sequence>>(8*i));
        block[pp.lt_offset+i]=uint8_t(lt>>(8*i));
    }
    block[pp.suffix_len]=0x80;
    uint64_t bits=uint64_t(pp.total_preimage_len)*8;
    for(unsigned i=0;i<8;i++)block[127-i]=uint8_t(bits>>(8*i));
    SHA256_CTX sc;SHA256_Init(&sc);for(unsigned i=0;i<8;i++)sc.h[i]=pp.midstate[i];
    SHA256_Transform(&sc,block);SHA256_Transform(&sc,block+64);
    uint8_t digest[32],zbytes[32];
    for(unsigned i=0;i<8;i++)for(unsigned j=0;j<4;j++)digest[4*i+j]=uint8_t(sc.h[i]>>(24-8*j));
    SHA256(digest,32,zbytes);
    BIGNUM* z=BN_bin2bn(zbytes,32,nullptr);BIGNUM* k=BN_new();BIGNUM* x=BN_new();BIGNUM* y=BN_new();
    EC_POINT* p=EC_POINT_new(gate.group);EC_POINT* q=EC_POINT_new(gate.group);
    EC_POINT* r=EC_POINT_dup(gate.recovery,gate.group);
    bool ok=z&&k&&x&&y&&p&&q&&r&&BN_mod_mul(k,z,gate.nri,gate.order,gate.ctx)==1&&
        EC_POINT_mul(gate.group,p,k,nullptr,nullptr,gate.ctx)==1;
    expected={};expected.usable=1;hit_bits=0;
    for(unsigned arm=0;arm<2&&ok;arm++) {
        if(arm)ok=EC_POINT_invert(gate.group,r,gate.ctx)==1;
        ok=ok&&EC_POINT_add(gate.group,q,p,r,gate.ctx)==1;
        if(!ok)break;
        if(EC_POINT_is_at_infinity(gate.group,q)==1){expected.usable=0;continue;}
        ok=EC_POINT_get_affine_coordinates(gate.group,q,x,y,gate.ctx)==1&&
            BN_bn2lebinpad(x,(uint8_t*)expected.x[arm],32)==32;
        if(!ok)break;
        uint32_t parity=BN_is_odd(y)?1u:0u;expected.parities|=parity<<arm;
        uint8_t pub[33],hash[32];pub[0]=uint8_t(2+parity);
        ok=BN_bn2binpad(x,pub+1,32)==32;
        if(ok){SHA256(pub,33,hash);if(qsb_host_zeros(hash)>=QSB_ZEROS_N)hit_bits|=1u<<arm;}
    }
    BN_free(z);BN_free(k);BN_free(x);BN_free(y);EC_POINT_free(p);EC_POINT_free(q);EC_POINT_free(r);
    return ok;
}

template<bool AUDIT>
static int qsb_affine_geometry(int device,unsigned& services,unsigned& workers,unsigned& grid) {
    int supported=0,blocks=0;cudaDeviceProp prop{};
    QSB_AFFINE_CUDA(cudaGetDeviceProperties(&prop,device));
    QSB_AFFINE_CUDA(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device));
    if(!supported){fprintf(stderr,"Affine search requires cooperative launch\n");return 2;}
    QSB_AFFINE_CUDA(cudaFuncSetAttribute(qsb_affine_search<AUDIT>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    QSB_AFFINE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,qsb_affine_search<AUDIT>,128,0));
    grid=unsigned(blocks*prop.multiProcessorCount);
    if(grid<2){fprintf(stderr,"Affine search has insufficient resident CTA capacity\n");return 2;}
    services=(grid+128u)/129u;workers=grid-services;
    if(!workers||services*128u<workers){fprintf(stderr,"Affine service capacity error\n");return 2;}
    printf("  Affine %s: %d CTAs/SM, %u workers, %u service CTAs, %u SMs\n",
           AUDIT?"startup audit":"search",blocks,workers,services,unsigned(prop.multiProcessorCount));
    return 0;
}

template<bool AUDIT>
static int qsb_affine_launch(const uint8_t* table,const QsbShiftedTailRecord* tails,
    qsb_inverse_service::Slot* slots,uint32_t* hit_mask,uint32_t* exception_mask,
    unsigned count,uint32_t start_lt,qsb_tail_pre tp,unsigned services,unsigned workers,unsigned grid,
    QsbAffineAudit* audit) {
    QSB_AFFINE_CUDA(cudaMemset(slots,0,workers*sizeof(*slots)));
    QSB_AFFINE_CUDA(cudaMemset(hit_mask,0,((size_t(count)+15)/16)*sizeof(uint32_t)));
    QSB_AFFINE_CUDA(cudaMemset(exception_mask,0,((size_t(count)+31)/32)*sizeof(uint32_t)));
    void* args[]={&table,&tails,&slots,&hit_mask,&exception_mask,&count,&start_lt,&tp,&services,&workers,&audit};
    QSB_AFFINE_CUDA(cudaLaunchCooperativeKernel((void*)qsb_affine_search<AUDIT>,grid,128,args));
    QSB_AFFINE_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int qsb_affine_driver(const pinning2_params_t& pp,const uint8_t* table,int device,
    int total_override,int global_offset,uint32_t sequence_override,uint32_t tail_w2,
    const timespec& process_started) {
    QsbAffineGate gate;
    if(!gate.init(pp)){fprintf(stderr,"Affine OpenSSL initialization failed\n");return 2;}
    unsigned services,workers,grid,audit_services,audit_workers,audit_grid;
    if(qsb_affine_geometry<false>(device,services,workers,grid)||
       qsb_affine_geometry<true>(device,audit_services,audit_workers,audit_grid))return 2;
    // A small audit grid makes 257 inputs cross a worker's tile boundary and
    // exercises a STOP mailbox while a neighboring lane still has requests.
    audit_workers=audit_grid>=3?2u:1u;audit_services=1;audit_grid=audit_workers+1;
    printf("  Affine audit uses %u worker CTAs to exercise tile reuse and partial STOP\n",audit_workers);
    std::vector<QsbShiftedTailRecord> tails(65536);
    if(!qsb_build_shifted_tail(tails.data(),pp.neg_r_inv,pp.u2r_x,pp.u2r_y)) {
        fprintf(stderr,"Affine shifted-tail construction failed\n");return 2;
    }
    QsbShiftedTailRecord* d_tails=nullptr;qsb_inverse_service::Slot* slots=nullptr;
    uint32_t *d_hits=nullptr,*d_exceptions=nullptr,*h_hits=nullptr,*h_exceptions=nullptr;
    QsbAffineAudit* d_audit=nullptr;
    const unsigned batch=QSB_BATCH;
    static_assert(QSB_BATCH>0&&QSB_BATCH<0x40000000,"affine batch must be a positive 30-bit count");
    const size_t hit_bytes=((size_t(batch)+15)/16)*4,exception_bytes=((size_t(batch)+31)/32)*4;
    QSB_AFFINE_CUDA(cudaMalloc(&d_tails,tails.size()*sizeof(tails[0])));
    QSB_AFFINE_CUDA(cudaMemcpy(d_tails,tails.data(),tails.size()*sizeof(tails[0]),cudaMemcpyHostToDevice));
    std::vector<QsbShiftedTailRecord>().swap(tails);
    QSB_AFFINE_CUDA(cudaMalloc(&slots,(workers>audit_workers?workers:audit_workers)*sizeof(*slots)));
    QSB_AFFINE_CUDA(cudaMalloc(&d_hits,hit_bytes));QSB_AFFINE_CUDA(cudaMalloc(&d_exceptions,exception_bytes));
    QSB_AFFINE_CUDA(cudaMallocHost(&h_hits,hit_bytes));QSB_AFFINE_CUDA(cudaMallocHost(&h_exceptions,exception_bytes));
    const unsigned audit_capacity=257;
    if(batch<audit_capacity){fprintf(stderr,"Affine batch must allow startup audit\n");return 2;}
    QSB_AFFINE_CUDA(cudaMalloc(&d_audit,audit_capacity*sizeof(*d_audit)));
    std::vector<QsbAffineAudit> observed(audit_capacity);
    const uint32_t audit_lt[3]={500000000u,0x0100fffeu,1744599743u};
    const unsigned audit_count[3]={257,129,255};
    unsigned checked=0;
    for(unsigned run=0;run<3;run++) {
        const uint32_t sequence=0x80000000u+17u*run;
        qsb_tail_pre tp=qsb_affine_sequence_params(pp,sequence,tail_w2);
        const unsigned count=audit_count[run];
        if(qsb_affine_launch<true>(table,d_tails,slots,d_hits,d_exceptions,count,audit_lt[run],tp,
            audit_services,audit_workers,audit_grid,d_audit))return 2;
        QSB_AFFINE_CUDA(cudaMemcpy(observed.data(),d_audit,count*sizeof(*d_audit),cudaMemcpyDeviceToHost));
        QSB_AFFINE_CUDA(cudaMemcpy(h_hits,d_hits,((count+15)/16)*4,cudaMemcpyDeviceToHost));
        QSB_AFFINE_CUDA(cudaMemcpy(h_exceptions,d_exceptions,((count+31)/32)*4,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<count;i++) {
            QsbAffineAudit expected{};uint32_t hits=0;
            if(!qsb_affine_reference(pp,sequence,audit_lt[run]+i,gate,expected,hits)){
                fprintf(stderr,"Affine startup OpenSSL oracle failed\n");return 2;
            }
            bool exception=((h_exceptions[i/32]>>(i%32))&1u)!=0;
            uint32_t got_hits=(h_hits[i/16]>>(2*(i%16)))&3u;
            // Exceptional candidates are safe on the real host cold path, but
            // these independently reconstructed ordinary fixtures must match.
            if(exception||!observed[i].usable||!expected.usable||
               memcmp(expected.x,observed[i].x,sizeof(expected.x))||
               expected.parities!=observed[i].parities||got_hits!=hits) {
                fprintf(stderr,"Affine startup audit FAILED run=%u candidate=%u exception=%u hits=%u/%u\n",
                    run,i,unsigned(exception),got_hits,hits);return 2;
            }
            checked++;
        }
    }
    cudaFree(d_audit);
    printf("  Affine startup GPU/OpenSSL audit passed: %u pairs, %u affine keys\n",checked,2*checked);
    fflush(stdout);
    int device_count=0;QSB_AFFINE_CUDA(cudaGetDeviceCount(&device_count));
    const unsigned total=unsigned(total_override>0?total_override:device_count);
    const unsigned effective_id=unsigned(global_offset+device);
    if(!total||effective_id>=total){fprintf(stderr,"Invalid affine GPU partition\n");return 2;}
    const uint32_t seq_min=sequence_override?sequence_override:0x80000000u;
    const uint32_t lt_min=500000000u,lt_max=1744600000u,lt_range=lt_max-lt_min;
    if(mkdir("results",0755)!=0&&errno!=EEXIST){perror("results");return 2;}
    char filename[128];snprintf(filename,sizeof(filename),"results/pinning_hit_%d.txt",device);
    uint64_t searched=0,published=0,exceptions=0;double last_progress=0;
    for(uint64_t seq=uint64_t(seq_min)+effective_id;seq<=UINT32_MAX;seq+=total) {
        qsb_tail_pre tp=qsb_affine_sequence_params(pp,uint32_t(seq),tail_w2);
        for(uint32_t offset=0;offset<lt_range;) {
            unsigned count=lt_range-offset;if(count>batch)count=batch;
            uint32_t start_lt=lt_min+offset;
            if(qsb_affine_launch<false>(table,d_tails,slots,d_hits,d_exceptions,count,start_lt,tp,
                services,workers,grid,nullptr))return 2;
            QSB_AFFINE_CUDA(cudaMemcpy(h_hits,d_hits,((size_t(count)+15)/16)*4,cudaMemcpyDeviceToHost));
            QSB_AFFINE_CUDA(cudaMemcpy(h_exceptions,d_exceptions,((size_t(count)+31)/32)*4,cudaMemcpyDeviceToHost));
            FILE* output=nullptr;
            for(unsigned base=0;base<count;base+=32) {
                const uint32_t exceptional=h_exceptions[base/32];
                const unsigned hw=base/16;
                const uint32_t hit0=h_hits[hw],hit1=base+16<count?h_hits[hw+1]:0;
                if(!(exceptional|hit0|hit1))continue;
                for(unsigned lane=0;lane<32&&base+lane<count;lane++) {
                    unsigned hit=(lane<16?hit0:hit1)>>(2*(lane&15));hit&=3u;
                    const bool fallback=((exceptional>>lane)&1u)!=0;
                    if(!hit&&!fallback)continue;
                    if(fallback)exceptions++;
                    uint32_t lt=start_lt+base+lane;
                    // Check both recids for every nominated or exceptional
                    // candidate; each unique sequence/locktime is visited once.
                    for(int ri=0;ri<2;ri++)if(qsb_host_exact_hit(&pp,uint32_t(seq),lt,ri,
                        gate.group,gate.ctx,gate.order,gate.nri,gate.recovery)) {
                        if(!output)output=fopen(filename,"a");
                        if(!output){perror(filename);return 2;}
                        if(fprintf(output,"sequence=%u locktime=%u recid=%d\n",uint32_t(seq),lt,ri)<0){perror(filename);fclose(output);return 2;}
                        published++;
                    }
                }
            }
            if(output&&fclose(output)!=0){perror(filename);return 2;}
            searched+=count;offset+=count;
            timespec now;clock_gettime(CLOCK_MONOTONIC,&now);
            double elapsed=(now.tv_sec-process_started.tv_sec)+(now.tv_nsec-process_started.tv_nsec)*1e-9;
            if(elapsed-last_progress>=1.0) {
                printf("  Affine (%lluM/%uM) %.3fM/s hits=%llu exceptions=%llu\n",
                    (unsigned long long)(searched/1000000),lt_range/1000000,
                    double(searched)/elapsed/1e6,(unsigned long long)published,(unsigned long long)exceptions);
                fflush(stdout);last_progress=elapsed;
            }
        }
    }
    fprintf(stderr,"Affine sequence partition exhausted without repetition\n");
    cudaFree(d_tails);cudaFree(slots);cudaFree(d_hits);cudaFree(d_exceptions);cudaFreeHost(h_hits);cudaFreeHost(h_exceptions);
    return 0;
}
#undef QSB_AFFINE_CUDA
