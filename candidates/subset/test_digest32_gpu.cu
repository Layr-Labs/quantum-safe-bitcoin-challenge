// SPDX-License-Identifier: GPL-3.0-only
// Run the actual candidate's SHA functions against OpenSSL on deterministic inputs.
#define main qsb_candidate_main
#include "tests/gpu_epochs/tree.cu"
#undef main

static void checked(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        exit(2);
    }
}

__global__ void compare_digest32(const uint32_t *inputs, uint32_t *outputs, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t m[8], generic[8], specialized[8], alias[8], block[16] = {};
    for (int j = 0; j < 8; ++j) m[j] = alias[j] = block[j] = inputs[i * 8 + j];
    block[8] = 0x80000000u;
    block[15] = 256;
    _SHA256Initialize(generic);
    _SHA256Transform(generic, block);
    _SHA256TransformDigest32(specialized, m);
    _SHA256TransformDigest32(alias, alias);
    uint64_t z[4];
    qsb_pair_second_sha_z(m, z);
    for (int j = 0; j < 8; ++j) {
        outputs[i * 32 + j] = generic[j];
        outputs[i * 32 + 8 + j] = specialized[j];
        outputs[i * 32 + 16 + j] = alias[j];
        outputs[i * 32 + 24 + j] = (uint32_t)(z[3 - j / 2] >> (j % 2 ? 0 : 32));
    }
}

int main() {
    const int count = 256 + 512 + 20000;
    uint32_t *inputs = (uint32_t *)malloc(count * 32);
    uint32_t *outputs = (uint32_t *)malloc(count * 128);
    uint32_t *expected = (uint32_t *)malloc(count * 32);
    if (!inputs || !outputs || !expected) return 2;
    uint64_t rng = 0xc75e25f4a319b62dULL;
    for (int i = 0; i < count; ++i) {
        unsigned char message[32], digest[32];
        if (i < 256) {
            memset(message, i, sizeof(message));
        } else if (i < 768) {
            int bit = (i - 256) % 256;
            memset(message, i < 512 ? 0 : 255, sizeof(message));
            message[bit / 8] ^= 1u << (bit % 8);
        } else {
            for (int j = 0; j < 32; ++j) {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                message[j] = (rng * 0x2545f4914f6cdd1dULL) >> 56;
            }
        }
        SHA256(message, sizeof(message), digest);
        for (int j = 0; j < 8; ++j) {
            inputs[i * 8 + j] = 0;
            expected[i * 8 + j] = 0;
            for (int k = 0; k < 4; ++k) {
                inputs[i * 8 + j] = (inputs[i * 8 + j] << 8) | message[4 * j + k];
                expected[i * 8 + j] = (expected[i * 8 + j] << 8) | digest[4 * j + k];
            }
        }
    }
    uint32_t *device_inputs, *device_outputs;
    checked(cudaMalloc(&device_inputs, count * 32), "allocate inputs");
    checked(cudaMalloc(&device_outputs, count * 128), "allocate outputs");
    checked(cudaMemcpy(device_inputs, inputs, count * 32, cudaMemcpyHostToDevice), "copy inputs");
    compare_digest32<<<(count + 255) / 256, 256>>>(device_inputs, device_outputs, count);
    checked(cudaGetLastError(), "launch");
    checked(cudaDeviceSynchronize(), "synchronize");
    checked(cudaMemcpy(outputs, device_outputs, count * 128, cudaMemcpyDeviceToHost), "copy outputs");
    for (int i = 0; i < count; ++i) {
        for (int implementation = 0; implementation < 4; ++implementation) {
            if (memcmp(outputs + i * 32 + implementation * 8, expected + i * 8, 32)) {
                fprintf(stderr, "FAIL vector=%d implementation=%d\n", i, implementation);
                return 1;
            }
        }
    }
    printf("PASS: %d inputs, %d full SHA-256 comparisons; generic, specialized, in-place, active z conversion vs OpenSSL\n", count, count * 4);
    checked(cudaFree(device_inputs), "free inputs");
    checked(cudaFree(device_outputs), "free outputs");
    free(inputs); free(outputs); free(expected);
    return 0;
}
