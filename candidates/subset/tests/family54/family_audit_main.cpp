static void read_exact(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) { fprintf(stderr,"short fixture\n"); exit(2); }
}
static bool audit_cpu_supported() {
    // Windows audit only: compiler-rt's Linux CPU-dispatch globals are absent.
    unsigned a,b,c,d;
    if(!__get_cpuid(1,&a,&b,&c,&d) || (c & (3u<<27)) != (3u<<27)) return false;
    unsigned lo,hi; __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if((lo & 6)!=6) return false;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) && (b & (1u<<29));
}
int main(int argc, char **argv) {
    if (argc != 2 || !audit_cpu_supported()) { fprintf(stderr,"need fixture and SHA-NI CPU\n"); return 2; }
    FILE *f = fopen(argv[1],"rb"); if (!f) return 2;
    uint32_t shape[2]; read_exact(f, shape, sizeof shape);
    if (shape[0] != CEARLY) return 3;
    unsigned total = 0;
    for (uint32_t trial=0; trial<shape[1]; ++trial) {
        uint8_t rows[1500], tail[218], suffix[44];
        read_exact(f,rows,sizeof rows); read_exact(f,tail,sizeof tail); read_exact(f,suffix,sizeof suffix);
        digest_params_t dp; dp.dummy_sigs=rows; dp.tail_section=tail; dp.tx_suffix=suffix;
        Ctx c; c.dp=&dp;
#if QSB_CPU_FAMILY54
        c.ncwin=qsb_family54::make(c.cwin);
#else
        for(int a=0;a<13;++a)for(int b=a+1;b<13;++b)for(int d=b+1;d<13;++d) {
            if(a>=6 || (a<=5 && b>=7) || (a==0 && b==1 && d>=8 && d<=10)) continue;
            c.cwin[c.ncwin][0]=137+a;c.cwin[c.ncwin][1]=137+b;c.cwin[c.ncwin][2]=137+d;++c.ncwin;
        }
#endif
        hash_plan_cpu_patterns(c);
        if(!c.hplan || c.ncwin!=(QSB_CPU_FAMILY54 ? 335:100) || c.h_ng!=(QSB_CPU_FAMILY54 ? 12:20) || c.h_nb!=6) return 4;
        // Real four-lane compression; pending lanes retain their epoch's buffer
        // across 335-pattern boundaries, as in the production worker.
        alignas(16) uint32_t states[2][CWIN_MAX+3][8];
        const uint32_t *lin[4]; const uint32_t *const *lrow[4];
        uint8_t expected[4][32]; int lane=0;
        for(int epoch=0;epoch<8;++epoch) {
            uint8_t early[CEARLY], rem[64]; uint32_t st[8];
            read_exact(f,early,sizeof early);read_exact(f,st,sizeof st);read_exact(f,rem,sizeof rem);
            std::vector<uint8_t> blocks(c.h_gblk);
            std::vector<uint32_t> wk((c.h_ng+3)*64);
            for(int g=0;g<c.h_ng;++g) {
                memcpy(&blocks[g*64],rem,(42+(137-CEARLY)*10)%64);
                qsha_schedule(&wk[g*64],&blocks[g*64]);
            }
            for(int g=c.h_ng;g<c.h_ng+3;++g)memcpy(&wk[g*64],wk.data(),256);
            auto gst=states[epoch&1];
            for(int g=0;g<c.h_ng;g+=4) {
                const uint32_t *gp[4];const uint32_t *const *gr[4]={&gp[0],&gp[1],&gp[2],&gp[3]};
                for(int l=0;l<4;++l){memcpy(gst[g+l],st,32);gp[l]=&wk[(g+l)*64];}
                qsha_x4p(&gst[g],gr,1);
            }
            for(int pi=0;pi<c.ncwin;++pi) {
                uint8_t want_skip[9],sk[10];read_exact(f,want_skip,9);read_exact(f,expected[lane],32);
                // Same packed identity writes as the worker, with canary after
                // nine bytes for FAMILY54 (legacy writes one byte into padding).
                memset(sk,0xA5,sizeof sk);uint64_t e8=0;memcpy(&e8,early,CEARLY);
                uint32_t cw4=0;memcpy(&cw4,c.cwin[pi],CWIN_BYTES);
                memcpy(sk,&e8,8);memcpy(sk+CEARLY,&cw4,4);
                if(memcmp(sk,want_skip,9) || (QSB_CPU_FAMILY54 && sk[9]!=0xA5)) return 5;
                lin[lane]=gst[c.h_g0[pi]];lrow[lane]=c.h_wkp[pi];++lane;
                if(lane==4) {
                    alignas(16) uint32_t w2[4][16]={},z[4][8];
                    for(int l=0;l<4;++l){w2[l][8]=0x80000000;w2[l][15]=256;}
                    qsha_x4p(nullptr,lrow,c.h_nb-1,&w2[0][0],16,lin);
                    qsha_x4w_iv32(z,w2);
                    for(int l=0;l<4;++l)for(int w=0;w<8;++w)for(int b=0;b<4;++b)
                        if((uint8_t)(z[l][w]>>(24-8*b))!=expected[l][4*w+b])return 6;
                    total+=4;lane=0;
                }
            }
        }
        if(lane) return 7;
    }
    if(fgetc(f)!=EOF)return 8;
    fclose(f);printf("PASS early=%d complete C++ SHA256d and identity checks=%u\n",CEARLY,total);
    return 0;
}
