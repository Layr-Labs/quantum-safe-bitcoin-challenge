#!/usr/bin/env python3
"""Materialize an isolated SHA/EC split experiment; never select or submit it.

The EC body is taken verbatim from the selected base. The promoted, submitted,
or pending base can be supplied with --base. Output must not already exist.
CUDA compilation is optional and is not GPU execution or a benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from preflight import source_files, source_identity


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Base source shape changed: expected one {old[:90]!r}")
    return text.replace(old, new)


def stage_source(tree):
    anchors = ("__global__ void __launch_bounds__(256, 2) kernel_digest(",
               "    /* EC recovery with both flags", "    /* neg_r_inv is folded",
               "    /* The bridge parses only")
    if any(tree.count(anchor) != 1 for anchor in anchors):
        raise ValueError("Unsupported source shape for split extraction")
    kernel_start, recovery_start, ec_start, hit_start = map(tree.index, anchors)
    second_start = tree.index("    uint32_t b2[16];", kernel_start)
    if not kernel_start < second_start < recovery_start < ec_start < hit_start:
        raise ValueError("Unexpected kernel section order")
    second = tree[second_start:recovery_start]
    ec = tree[ec_start:hit_start]
    helpers = r'''
// Generated experiment: field/recovery code retains the selected base's notices.
// Exactly one complete 256-lane epoch per block; partial epochs are unsupported.
__device__ __forceinline__ int qsb_split_index(int epoch_offset, int local_idx) {
    return epoch_offset * QSB_SE_PER_EPOCH + local_idx;
}
__device__ __forceinline__ void qsb_split_load_z(const uint32_t *hashes,
        int stride, int local_idx, uint64_t *z) {
    #pragma unroll
    for (int k=0;k<4;k++)
        z[k]=((uint64_t)hashes[(6-2*k)*stride+local_idx]<<32)
             | hashes[(7-2*k)*stride+local_idx];
}
__device__ __forceinline__ void qsb_split_combo(uint8_t *out,
        const epoch_desc_t *epoch, int lane) {
    for (int i=0;i<6;i++) out[i]=epoch->early[i];
    for (int i=0;i<3;i++) out[6+i]=WIN3[lane][i];
}
__global__ void __launch_bounds__(256) qsb_split_hash(
        const epoch_desc_t *epochs, uint32_t *hashes) {
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=gridDim.x*QSB_SE_PER_EPOCH;
    uint32_t state[8];
    qsb_scheduled_window_hash(state,epochs+blockIdx.x,threadIdx.x);
'''
    helpers += second + r'''
    #pragma unroll
    for (int k=0;k<8;k++) hashes[k*stride+idx]=s2[k];
}
'''
    consumer = r'''
__global__ void __launch_bounds__(256, 2) qsb_split_ec(
        const uint32_t *hashes, int epoch_offset, const epoch_desc_t *epochs,
        const uint64_t *d_u2rx, const uint64_t *d_u2ry,
        uint8_t *d_gtX, uint8_t *d_gtY,
        uint32_t *d_hit_cnt, uint32_t *d_hit_idx, uint8_t *d_hit_combos,
        int easy_mode, int single_hash, int calibrate_mode) {
    const int local_idx=blockIdx.x*blockDim.x+threadIdx.x;
    const int idx=qsb_split_index(epoch_offset,local_idx);
    const int active=1; // the host submits only whole epochs, including the last tile
    uint64_t z[4];
    qsb_split_load_z(hashes,gridDim.x*QSB_SE_PER_EPOCH,local_idx,z);
'''
    if "template <bool RankedShortEpoch>" in tree:
        # This experiment replaces only the guarded, ranked short-epoch launch.
        consumer += "    const bool RankedShortEpoch = true;\n"
    consumer += ec + r'''
    if (v) {
        uint32_t p=atomicAdd(d_hit_cnt,1);
        if (p<1024) {
            d_hit_idx[p]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);
            qsb_split_combo(d_hit_combos+p*MAX_T,
                            epochs+epoch_offset+blockIdx.x,threadIdx.x);
        }
    }
}
'''
    return helpers, consumer


def transform(tree):
    for name,value in (("N_INC",10),("EARLY",6),("TWIN",3),("CUT",137),("PER_EPOCH",256)):
        if len(re.findall(rf"^#define\s+QSB_SE_{name}\s+{value}\b",tree,re.M)) != 1:
            raise ValueError("Only the inspected 256-lane short-epoch shape is supported")
    header = """// Isolated research build. Zero selects the unchanged monolithic control.
#ifndef QSB_SPLIT_TILE_EPOCHS
#define QSB_SPLIT_TILE_EPOCHS 256
#endif
#if QSB_SPLIT_TILE_EPOCHS < 0 || QSB_SPLIT_TILE_EPOCHS > 32768
#error Invalid split tile size
#endif
"""
    result = header + tree
    anchor = "/* ============================================================\n * Fixed-base table construction on the GPU (signed-digit table)"
    result = once(result, anchor, '#if QSB_SPLIT_TILE_EPOCHS > 0\n#include "split_stage.cuh"\n#endif\n\n' + anchor)
    result = once(result, "    epoch_desc_t *d_epochs = NULL;",
                  "    epoch_desc_t *d_epochs = NULL;\n#if QSB_SPLIT_TILE_EPOCHS > 0\n    uint32_t *d_split_hashes = NULL;\n#endif")
    anchor = '        if (!d_epochs) { fprintf(stderr, "OOM: epoch descriptors\\n"); return 1; }'
    result = once(result, anchor, anchor + r'''
#if QSB_SPLIT_TILE_EPOCHS > 0
        if(cudaMalloc(&d_split_hashes,(size_t)QSB_SPLIT_TILE_EPOCHS*256*32)!=cudaSuccess) {
            fprintf(stderr,"OOM: split digest tile\n"); return 1;
        }
#endif''')
    short_start = result.index("    /* Short-epoch path: producer/consumer on GPU")
    short_end = result.index("        free(h_combos);", short_start)
    specialization = "<true>" if "template <bool RankedShortEpoch>" in tree else ""
    launch = f"            kernel_digest{specialization}<<<nblk, QSB_SE_PER_EPOCH>>>("
    if result.count(launch) != 1:
        raise ValueError("Expected exactly one short-epoch consumer launch")
    start = result.index(launch,short_start)
    end = result.index("            cudaDeviceSynchronize();", start)
    if not short_start < start < end < short_end:
        raise ValueError("Consumer launch is outside the inspected short-epoch branch")
    if "        return 0;" not in result[short_end:short_end+90]:
        raise ValueError("Short-epoch branch must exit before legacy launch paths")
    original_launch = result[start:end]
    split_launch = r'''
#if QSB_SPLIT_TILE_EPOCHS > 0
            for(int offset=0;offset<nblk;offset+=QSB_SPLIT_TILE_EPOCHS) {
                int tile=nblk-offset;
                if(tile>QSB_SPLIT_TILE_EPOCHS)tile=QSB_SPLIT_TILE_EPOCHS;
                // Same stream orders producer -> hash -> EC -> buffer reuse.
                qsb_split_hash<<<tile,256>>>(d_epochs+offset,d_split_hashes);
                if(cudaGetLastError()!=cudaSuccess)return 1;
                qsb_split_ec<<<tile,256>>>(d_split_hashes,offset,d_epochs,
                    d_u2rx,d_u2ry,d_gtX,d_gtY,d_hit_cnt,d_hit_idx,d_hit_combos,
                    easy,single_hash,calibrate);
                if(cudaGetLastError()!=cudaSuccess)return 1;
            }
#else
''' + original_launch + "#endif\n"
    result = result[:start] + split_launch + result[end:]
    # This is the short-epoch path's successful exit; fatal errors exit the process.
    short_start = result.index("    /* Short-epoch path: producer/consumer on GPU")
    short_end = result.index("        free(h_combos);", short_start)
    result = (result[:short_end] + "#if QSB_SPLIT_TILE_EPOCHS > 0\n"
              "        cudaFree(d_split_hashes);\n#endif\n" + result[short_end:])
    return result


def materialize(base, output):
    base = base.resolve()
    if output.exists():
        raise ValueError("Output already exists; use a fresh experiment directory")
    tree = (base / "tests/gpu_epochs/tree.cu").read_text()
    helpers, consumer = stage_source(tree)
    modified = transform(tree)  # fail before writing if the base changed
    identity = source_identity(base)
    output.mkdir(parents=True)
    for path in source_files(base):
        dest = output / path.relative_to(base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    if (base / "COPYING").exists():
        shutil.copyfile(base / "COPYING", output / "COPYING")
    (output / "tests/gpu_epochs/tree.cu").write_text(modified)
    (output / "tests/gpu_epochs/split_stage.cuh").write_text(helpers + consumer)
    return {"base": identity, "experiment": source_identity(output),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_compile": "not_run", "gpu_executed": False,
            "performance": "unknown", "selected_for_submission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true", help="compile control plus three tile sizes")
    args = parser.parse_args()
    report = materialize(args.base, args.output)
    if args.cuda:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            report["cuda_compile"] = "unavailable"
            (args.output / "experiment.json").write_text(json.dumps(report, indent=2)+"\n")
            parser.error("nvcc unavailable; experiment generated but not compiled")
        report["compiler"] = subprocess.check_output([nvcc,"--version"],text=True)
        for tile in (0,128,256,1024):
            command = [nvcc,"-O3","-arch=sm_89","-Xptxas=-v","-DQSB_ZEROS_N=24",
                       f"-DQSB_SPLIT_TILE_EPOCHS={tile}",str(args.output/"subset.cu"),
                       "-o",str(args.output/f"subset-tile-{tile}"),"-lcrypto","-lm"]
            with (args.output/f"compile-{tile}.log").open("w") as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        report["cuda_compile"] = "pass_control_and_128_256_1024_epochs"
    (args.output / "experiment.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
