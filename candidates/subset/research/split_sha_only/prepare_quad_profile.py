"""Host-only event instrumentation for the prepared four-stage candidate."""
from pathlib import Path
import os
import re

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
tree = ROOT / "candidates/subset/tests/gpu_epochs/tree.cu"
s = tree.read_text()
s = re.sub(r'^#include "([^"]+)"', lambda m: '#include "' + os.path.relpath(
    (tree.parent / m.group(1)).resolve(), P) + '"', s, flags=re.M)
wrapper = (ROOT / "candidates/subset/subset.cu").read_text()
s = wrapper.replace('#include "tests/gpu_epochs/tree.cu"', s)
s = '#define QSB_PROF_CUDA(call) do { cudaError_t pe=(call); if(pe!=cudaSuccess){fprintf(stderr,"profile CUDA error: %s\\n",cudaGetErrorString(pe));return 2;} } while(0)\n' + s


def add_before(token, addition):
    global s
    assert token in s, token
    s = s.replace(token, addition + token, 1)


setup = r'''
        cudaDeviceProp qsb_prop;
        QSB_PROF_CUDA(cudaGetDeviceProperties(&qsb_prop,gpu_index));
        printf("PROFILE_DEVICE name=%s cc=%d.%d sms=%d l2_bytes=%d\n",qsb_prop.name,qsb_prop.major,qsb_prop.minor,qsb_prop.multiProcessorCount,qsb_prop.l2CacheSize);
        cudaFuncAttributes qsb_attr; int qsb_active=0;
'''
for label, name in (("sha", "kernel_split_sha"), ("point", "kernel_split_point"),
                    ("inverse", "kernel_split_inverse4"), ("tail", "kernel_split_tail")):
    setup += f'''
        QSB_PROF_CUDA(cudaFuncGetAttributes(&qsb_attr,{name}));
        QSB_PROF_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qsb_active,{name},256,0));
        printf("PROFILE_ATTR kernel={label} registers=%d local_bytes=%zu shared_bytes=%zu active_blocks=%d\\n",qsb_attr.numRegs,qsb_attr.localSizeBytes,qsb_attr.sharedSizeBytes,qsb_active);
'''
setup += '''
        cudaEvent_t qsb_events[8];
        for(int i=0;i<8;i++)QSB_PROF_CUDA(cudaEventCreate(&qsb_events[i]));
        unsigned qsb_profile_batch=0;
'''
add_before("        while (1) {", setup)
s = s.replace("        while (1) {", "        while (1) {\n            QSB_PROF_CUDA(cudaEventRecord(qsb_events[0]));", 1)
for index, token in (
    (1, "            // One producer block for each valid epoch"),
    (2, "            kernel_split_sha<<<"),
    (3, "            kernel_split_point<<<"),
    (4, "            kernel_split_inverse4<<<"),
    (5, "            kernel_split_tail<<<"),
    (6, "            kernel_verify_pair_hits<<<"),
    (7, "            // Blocking hit-buffer copy below"),
):
    add_before(token, f"            QSB_PROF_CUDA(cudaEventRecord(qsb_events[{index}]));\n")
add_before("            // Publish only completed batches", r'''
            float qsb_ms[7];
            for(int i=0;i<7;i++)QSB_PROF_CUDA(cudaEventElapsedTime(&qsb_ms[i],qsb_events[i],qsb_events[i+1]));
            printf("PROFILE batch=%u producer=%.6f first=%.6f sha=%.6f point=%.6f inverse=%.6f tail=%.6f replay=%.6f ms\n",++qsb_profile_batch,qsb_ms[0],qsb_ms[1],qsb_ms[2],qsb_ms[3],qsb_ms[4],qsb_ms[5],qsb_ms[6]);
            fflush(stdout);
            if(qsb_profile_batch==24){
                for(int i=0;i<8;i++)QSB_PROF_CUDA(cudaEventDestroy(qsb_events[i]));
                return 0; // Unscored diagnostic only.
            }
''')
(P / "profile_quad.cu").write_text(s)
