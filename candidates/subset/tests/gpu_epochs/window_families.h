#pragma once
#include <stdint.h>
#include <string.h>

/* Host-only selection of disjoint 3-of-13 suffix families. The six earlier
 * omissions are independent of these suffix triples. phase 0 preserves the
 * promoted lane order; phase 1 consumes 128 previously unused triples. */
static uint32_t qsb_family_first_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=0,n=0;i<13 && n<6;i++)
        if(i!=w[0] && i!=w[1] && i!=w[2]) { key=(key<<4)|(uint32_t)i; n++; }
    return key;
}
static uint32_t qsb_family_second_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=12,n=0;i>=0 && n<5;i--)
        if(i!=w[0] && i!=w[1] && i!=w[2]) { key=(key<<4)|(uint32_t)i; n++; }
    return key;
}
static int qsb_family_in_a(int a,int b,int c) {
    return a>=6 || (a<=5 && b>=7) || (a==0 && b==1 && c>=8 && c<=10);
}
static int qsb_select_window_family(uint8_t (*windows)[3],int count,int phase) {
    if(!windows || (count!=128 && count!=256) || phase<0 || phase>1 ||
       (count==256 && phase!=0)) return 1;
    uint8_t pool[286][3];
    uint32_t key[286];
    unsigned sizes[286];
    int n=0;
    for(int a=0;a<13;a++) for(int b=a+1;b<13;b++) for(int c=b+1;c<13;c++) {
        const int in_a=qsb_family_in_a(a,b,c);
        if(count==256) {
            if(a>=1 && c<=7 && !(a==1 && b==2)) continue;
        } else if((phase==0 && !in_a) || (phase==1 && in_a)) continue;
        pool[n][0]=(uint8_t)a; pool[n][1]=(uint8_t)b; pool[n][2]=(uint8_t)c;
        key[n]=qsb_family_first_key(pool[n]);
        n++;
    }
    if(phase==1) {
        /* Select whole largest first-state classes first. Lexicographic first
         * keys break equal-size ties. Initial lex triple order breaks ties
         * inside a class; insertion sort is stable. */
        for(int i=0;i<n;i++) {
            sizes[i]=0;
            for(int j=0;j<n;j++) sizes[i]+=(key[i]==key[j]);
        }
        for(int i=1;i<n;i++) {
            uint8_t w[3]; memcpy(w,pool[i],3);
            const uint32_t k=key[i]; const unsigned s=sizes[i]; int j=i;
            while(j>0 && (sizes[j-1]<s || (sizes[j-1]==s && key[j-1]>k))) {
                memcpy(pool[j],pool[j-1],3); key[j]=key[j-1]; sizes[j]=sizes[j-1]; j--;
            }
            memcpy(pool[j],w,3); key[j]=k; sizes[j]=s;
        }
    }
    if(n<count || (phase==0 && n!=count)) return 1;
    /* Match the promoted second-block/first-block lane grouping exactly. */
    for(int i=1;i<count;i++) {
        uint8_t w[3]; memcpy(w,pool[i],3);
        const uint32_t second=qsb_family_second_key(w),first=qsb_family_first_key(w);
        int j=i;
        while(j>0 && (qsb_family_second_key(pool[j-1])>second ||
              (qsb_family_second_key(pool[j-1])==second && qsb_family_first_key(pool[j-1])>first))) {
            memcpy(pool[j],pool[j-1],3); j--;
        }
        memcpy(pool[j],w,3);
    }
    for(int i=0;i<count;i++) for(int j=0;j<3;j++) windows[i][j]=(uint8_t)(137+pool[i][j]);
    return 0;
}
