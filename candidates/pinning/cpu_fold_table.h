/* Build signed windows and the two folded last-window banks at startup. */
#pragma once
static uint64_t cg_read_memory(const char *path,uint64_t fallback) {
    FILE *f=fopen(path,"r");if(!f)return fallback;
    char text[64]={};const int n=fscanf(f,"%63s",text);fclose(f);
    if(n!=1 || !strcmp(text,"max"))return fallback;
    char *end=NULL;const unsigned long long v=strtoull(text,&end,10);
    return end && !*end?(uint64_t)v:fallback;
}
static bool signed_memory_ok(int nw) {
    const uint64_t unlimited=~0ULL;
    const long pages=sysconf(_SC_AVPHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
    uint64_t avail=pages>0 && page_size>0?(uint64_t)pages*(uint64_t)page_size:0;
    uint64_t limit=cg_read_memory("/sys/fs/cgroup/memory.max",unlimited);
    uint64_t used=cg_read_memory("/sys/fs/cgroup/memory.current",unlimited);
    if(limit==unlimited) {
        limit=cg_read_memory("/sys/fs/cgroup/memory/memory.limit_in_bytes",unlimited);
        used=cg_read_memory("/sys/fs/cgroup/memory/memory.usage_in_bytes",unlimited);
    }
    if(limit!=unlimited && used!=unlimited) {
        const uint64_t headroom=limit>used?limit-used:0;
        if(headroom<avail)avail=headroom;
    }
    /* Legacy 64 MiB table is already allocated; leave at least 512 MiB plus
     * two MiB per worker for all vector/scalar states and other host activity. */
    return avail>=qcg_signed_plan::table_bytes+(512ull<<20)+(uint64_t)nw*(2ull<<20);
}
static void fold_last_window(tentry *plus,tentry *minus,unsigned count,const fe &ax,const fe &ay) {
    tentry_set(plus,ax,ay);fe nay;fe_neg(&nay,&ay,1);fe_normalize(&nay);tentry_set(minus,ax,nay);
    for(unsigned begin=1;begin<count;begin+=256) {
        const int take=(int)(count-begin<256?count-begin:256);
        fe x[256],y[256],dx[256],prefix[256],inv,t;bool same_x[256];
        for(int k=0;k<take;++k) {
            fe_from_w(&x[k],plus[begin+k].x);fe_from_w(&y[k],plus[begin+k].y);
            fe_neg(&t,&x[k],1);dx[k]=ax;fe_add(&dx[k],&t);fe_normalize(&dx[k]);
            same_x[k]=fe_is_zero_norm(&dx[k]);if(same_x[k])fe_one(&dx[k]);
            if(k)fe_mul(&prefix[k],&prefix[k-1],&dx[k]);else prefix[k]=dx[k];
        }
        fe_normalize(&prefix[take-1]);fe_inv(&inv,&prefix[take-1]);
        for(int k=take-1;k>=0;--k) {
            fe ik;if(k)fe_mul(&ik,&inv,&prefix[k-1]);else ik=inv;
            fe_mul(&inv,&inv,&dx[k]);
            for(int r=0;r<2;++r) {
                tentry *out=(r?minus:plus)+begin+k;const fe &ty=r?nay:ay;
                if(same_x[k]) {
                    fe delta;fe_neg(&t,&y[k],1);delta=ty;fe_add(&delta,&t);fe_normalize(&delta);
                    if(fe_is_zero_norm(&delta)) {fe xx=x[k],yy=y[k];aff_dbl1(&xx,&yy);tentry_set(out,xx,yy);}
                    else {memset(out,0,sizeof *out);g_cg->signed_infinity[r]=true;}
                } else {
                    fe dy,lambda,qx,qy,tx;
                    fe_neg(&t,&y[k],1);dy=ty;fe_add(&dy,&t);fe_mul(&lambda,&dy,&ik);
                    fe_sqr(&qx,&lambda);fe_neg(&t,&x[k],1);fe_add(&qx,&t);fe_neg(&t,&ax,1);fe_add(&qx,&t);fe_normalize_weak(&qx);
                    fe_neg(&t,&qx,1);tx=x[k];fe_add(&tx,&t);fe_mul(&qy,&lambda,&tx);
                    fe_neg(&t,&y[k],1);fe_add(&qy,&t);fe_normalize_weak(&qy);tentry_set(out,qx,qy);
                }
            }
        }
    }
}
static void build_signed_window(int j,const fe &bx,const fe &by) {
    namespace plan=qcg_signed_plan;
    tentry *e=g_cg->signed_table+plan::offset(j);
    memset(e,0,sizeof *e);build_window(e,&bx,&by,(unsigned)plan::rows(j));
    if(j==plan::windows-1)fold_last_window(e,e+plan::rows(j),(unsigned)plan::rows(j),g_cg->ax,g_cg->ay);
}
