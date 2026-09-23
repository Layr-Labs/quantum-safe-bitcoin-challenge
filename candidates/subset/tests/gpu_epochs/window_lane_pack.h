// Same128omission windows, packed as complete second-message classes.
#pragma once
#include <stdint.h>
#include <string.h>
static int qsb_pack_128_window_lanes(uint8_t windows[128][3]) {
    // Verify the exact source ordering before using the fixed permutation.
    // Any different geometry keeps the caller's original order untouched.
    static const uint16_t expected[128]={
        3258,3257,3256,3255,3254,3253,3252,3251,3250,3249,3248,3241,3240,3239,3238,3237,
        3236,3235,3234,3233,3232,3224,3223,3222,3221,3220,3219,3218,3217,3216,3207,3206,
        3205,3204,3203,3202,3201,3200,3190,3189,3188,3187,3186,3185,3184,2985,2984,2983,
        2982,2981,2980,2979,2978,2977,2976,2968,2967,2966,2965,2964,2963,2962,2961,2960,
        2951,2950,2949,2948,2947,2946,2945,2944,2934,2933,2932,2931,2930,2929,2928,2712,
        2711,2710,2709,2708,2707,2706,2705,2704,2695,2694,2693,2692,2691,2690,2689,2688,
        2678,2677,2676,2675,2674,2673,2672,2576,2439,2438,2437,2436,2435,2434,2433,2432,
        2422,2421,2420,2419,2418,2417,2416,2320,2166,2165,2164,2163,2162,2161,2160,2064
    };
    static const uint8_t permutation[128]={
        5,6,7,8,9,10,15,16,17,18,19,20,24,25,26,27,
        28,29,32,33,34,35,36,37,0,1,2,3,4,11,12,13,
        39,40,41,42,43,44,49,50,51,52,53,54,58,59,60,61,
        62,63,66,67,68,69,70,71,14,21,22,23,30,31,38,45,
        73,74,75,76,77,78,82,83,84,85,86,87,90,91,92,93,
        94,95,97,98,99,100,101,102,46,47,48,55,56,57,64,65,
        106,107,108,109,110,111,113,114,115,116,117,118,121,122,123,124,
        125,126,72,79,80,81,88,89,96,104,105,112,120,103,119,127
    };
    uint8_t reordered[128][3];
    unsigned seen[4]={0,0,0,0};
    for(unsigned i=0;i<128;i++) {
        const unsigned a=(unsigned)windows[i][0]-137u;
        const unsigned b=(unsigned)windows[i][1]-137u;
        const unsigned c=(unsigned)windows[i][2]-137u;
        if(a>=13 || b>=13 || c>=13 || !(a<b && b<c) ||
           (a|(b<<4)|(c<<8))!=expected[i]) return 0;
        const unsigned p=permutation[i];
        const unsigned bit=1u<<(p&31u);
        if(p>=128 || (seen[p>>5]&bit)) return 0;
        seen[p>>5]|=bit;
        memcpy(reordered[i],windows[p],3);
    }
    memcpy(windows,reordered,sizeof(reordered));
    return 1;
}
