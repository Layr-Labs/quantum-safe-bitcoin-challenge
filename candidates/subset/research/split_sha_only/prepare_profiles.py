"""Add host-only timing and driver resource queries; preserve device PTX."""
from pathlib import Path
import tarfile

P = Path(__file__).resolve().parent

CHECK = r'''
#define QSB_PROF_CUDA(call) do { cudaError_t qsb_pe=(call); if(qsb_pe!=cudaSuccess){fprintf(stderr,"profile CUDA error: %s\n",cudaGetErrorString(qsb_pe));return 2;} } while(0)
'''

SETUP = r'''
        cudaDeviceProp qsb_prop;
        QSB_PROF_CUDA(cudaGetDeviceProperties(&qsb_prop,gpu_index));
        printf("PROFILE_DEVICE name=%s cc=%d.%d sms=%d l2_bytes=%d\n",qsb_prop.name,qsb_prop.major,qsb_prop.minor,qsb_prop.multiProcessorCount,qsb_prop.l2CacheSize);
        cudaFuncAttributes qsb_attr; int qsb_active=0;
        QSB_PROF_CUDA(cudaFuncGetAttributes(&qsb_attr,kernel_sha_only));
        QSB_PROF_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qsb_active,kernel_sha_only,256,0));
        printf("PROFILE_ATTR kernel=sha registers=%d local_bytes=%zu shared_bytes=%zu active_blocks=%d\n",qsb_attr.numRegs,qsb_attr.localSizeBytes,qsb_attr.sharedSizeBytes,qsb_active);
        QSB_PROF_CUDA(cudaFuncGetAttributes(&qsb_attr,kernel_digest));
        QSB_PROF_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qsb_active,kernel_digest,QSB_SE_BLOCK,0));
        printf("PROFILE_ATTR kernel=consumer registers=%d local_bytes=%zu shared_bytes=%zu active_blocks=%d\n",qsb_attr.numRegs,qsb_attr.localSizeBytes,qsb_attr.sharedSizeBytes,qsb_active);
        cudaEvent_t qsb_events[6];
        for(int i=0;i<6;i++)QSB_PROF_CUDA(cudaEventCreate(&qsb_events[i]));
        unsigned qsb_profile_batch=0;
'''

REPORT = r'''
            float qsb_ms[5];
            for(int i=0;i<5;i++)QSB_PROF_CUDA(cudaEventElapsedTime(&qsb_ms[i],qsb_events[i],qsb_events[i+1]));
            printf("PROFILE batch=%u producer=%.6f first=%.6f sha=%.6f consumer=%.6f replay=%.6f ms\n",++qsb_profile_batch,qsb_ms[0],qsb_ms[1],qsb_ms[2],qsb_ms[3],qsb_ms[4]);
            fflush(stdout);
            if(qsb_profile_batch==24){
                for(int i=0;i<6;i++)QSB_PROF_CUDA(cudaEventDestroy(qsb_events[i]));
                return 0; // Diagnostic only: not a ranked or scored run.
            }
'''


def replace_once(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


for label, source in (("paired", "candidate.cu"), ("bound3", "candidate_bound3.cu"),
                      ("single", "candidate_single.cu")):
    s = CHECK + (P / source).read_text()
    s = replace_once(s, "        while (1) {", SETUP + "        while (1) {\n            QSB_PROF_CUDA(cudaEventRecord(qsb_events[0]));")
    for index, token in (
        (1, "            // One producer block for each valid epoch"),
        (2, "            const unsigned sha_count="),
        (3, "            kernel_digest<<<nblk, QSB_SE_BLOCK>>>("),
        (4, "            kernel_verify_pair_hits<<<1,64>>>("),
    ):
        s = replace_once(s, token, f"            QSB_PROF_CUDA(cudaEventRecord(qsb_events[{index}]));\n" + token)
    s = replace_once(s, "            // Blocking hit-buffer copy below", "            QSB_PROF_CUDA(cudaEventRecord(qsb_events[5]));\n            // Blocking hit-buffer copy below")
    s = replace_once(s, "            // Publish only completed batches", REPORT + "            // Publish only completed batches")
    (P / f"profile_{label}.cu").write_text(s)

with tarfile.open(P / "closed-through-single.tar.xz", "r:xz") as archive:
    data = archive.extractfile("output-candidate_single-j/problems/subset.bin").read()
(P / "profile-problem.bin").write_bytes(data)
