// GPL-derived PR137 helper; IvanLudvig, unpromoted source 103a6adc15391e07aac210a2653f8dfd7eb4d4c0
#include <stdint.h>
#include <cstring>
static uint32_t qsb_window_second_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=12,n=0;i>=0 && n<5;i--)
        if(i!=w[0]-137 && i!=w[1]-137 && i!=w[2]-137){key=(key<<4)|i;n++;}
    return key;
}
static int qsb_pack_second_classes(uint8_t windows[256][3]) {
    uint32_t keys[256];
    int counts[256]={0},nclass=0;
    for(int lane=0;lane<256;lane++){
        uint32_t key=qsb_window_second_key(windows[lane]);
        int slot=0;
        while(slot<nclass && keys[slot]!=key)slot++;
        if(slot==nclass){keys[nclass]=key;nclass++;}
        counts[slot]++;
    }
    for(int i=1;i<nclass;i++){
        uint32_t key=keys[i];int count=counts[i],j=i;
        while(j>0 && (counts[j-1]<count ||
              (counts[j-1]==count && keys[j-1]>key))){
            keys[j]=keys[j-1];counts[j]=counts[j-1];j--;
        }
        keys[j]=key;counts[j]=count;
    }
    uint8_t packed[256][3];
    int fill[8]={0};
    for(int slot=0;slot<nclass;slot++){
        int warp=0;
        while(warp<8 && fill[warp]+counts[slot]>32)warp++;
        if(warp==8)return 1;
        for(int lane=0;lane<256;lane++)
            if(qsb_window_second_key(windows[lane])==keys[slot]){
                memcpy(packed[warp*32+fill[warp]],windows[lane],3);
                fill[warp]++;
            }
    }
    for(int warp=0;warp<8;warp++)if(fill[warp]!=32)return 1;
    memcpy(windows,packed,sizeof(packed));
    return 0;
}