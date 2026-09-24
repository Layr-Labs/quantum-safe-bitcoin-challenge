// Diagnostic hypothesis: smaller producer/consumer batches reduce working-set
// size and the amount of completed work withheld by the final timed-out batch.
// Arithmetic, enumeration, verification, and hit encoding remain unchanged.
#define ZLAB_LAUNCH_BLOCKS 32768
#include "../../subset.cu"
