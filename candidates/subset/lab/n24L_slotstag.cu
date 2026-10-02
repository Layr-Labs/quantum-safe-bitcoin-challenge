// Hardware-slot tail phasing across four resident 128-thread blocks.
#define QSB_LOCAL_SM86 1
#define QSB_SE_BLOCK 128
#define ZLAB_LAUNCH_BLOCKS 524288
#define QSB_TAIL_STAGGER 4
#define QSB_PARK128 0
#include "../subset.cu"
