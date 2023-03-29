#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdbool.h>

#include "sha256.cuh"
#include "../requests.cuh"
#include "../crypto/hex.cuh"

#define TOTAL_SIZE 108
#define MAX_SHARES 16

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__ __constant__ unsigned char prefix_c[TOTAL_SIZE-4];
__device__ __constant__ char share_chunk_c[64];
__device__ __constant__ size_t share_difficulty_c;

__device__ __forceinline__ void sha256_to_hex(unsigned char *hash, char *hex) {
    static const char digits[] = "0123456789abcdef";

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        char lo_nibble = digits[hash[i] & 0x0F];
        char hi_nibble = digits[(hash[i] & 0xF0) >> 4];
        *hex++ = hi_nibble;
        *hex++ = lo_nibble;
    }
    *hex = '\0';
}

__device__ __forceinline__ bool is_valid(const char* str) {
    int mask = 0;

#pragma unroll
    for (int i = 0; i < share_difficulty_c; ++i) {
        mask |= (str[i] ^ share_chunk_c[i]);
    }
    return mask == 0;
}

__global__ void miner(unsigned char **out, int *stop, int *share_id) {
    const __restrict__ uint32_t tid = threadIdx.x;

    __shared__ SHA256_CTX prefix_ctx;
    if (tid == 0) {
        sha256_init_dev(&prefix_ctx);
        sha256_update_dev(&prefix_ctx, prefix_c, sizeof(unsigned char) * (TOTAL_SIZE-4));
    }
    __syncthreads();

    unsigned char _hex[TOTAL_SIZE];
    memcpy(_hex, prefix_c, sizeof(unsigned char) * (TOTAL_SIZE-4));

    SHA256_CTX ctx;
    unsigned char hash[32];
    char hash_hex[64];

    for (uint32_t index = blockIdx.x * blockDim.x + tid; *stop != 1; index += blockDim.x * gridDim.x) {
        _hex[TOTAL_SIZE-1] = index;
        _hex[TOTAL_SIZE-2] = index >> 8;
        _hex[TOTAL_SIZE-3] = index >> 16;
        _hex[TOTAL_SIZE-4] = index >> 24;

        ctx = prefix_ctx;

        sha256_update_dev(&ctx, _hex + (TOTAL_SIZE-4), sizeof(unsigned char) * 4);
        sha256_final_dev(&ctx, hash);
        sha256_to_hex(hash, hash_hex);

        if (is_valid(hash_hex)) {
            int id = atomicAdd(share_id, 1);
            memcpy(out[id], _hex, sizeof(unsigned char) * TOTAL_SIZE);

            if (id >= MAX_SHARES-2) {
                *stop = 1;
            }
        }

        if (index >= 0xFFFFFFFF) {
            *stop = 1;
        }
    }
}

void start(const int device_id, const int threads, const int blocks, unsigned char prefix[TOTAL_SIZE-4], char *share_chunk, size_t share_difficulty, const char *poolUrl, const char pending_transactions_hashes[512][64 + 1], size_t pending_transactions_count, uint id, bool silent) {
    auto res = cudaSetDevice(device_id);
    if (res != cudaSuccess) {
        printf("Error setting device: %s\n", cudaGetErrorString(res));
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));

    // allocate memory on the device
    int zero = 0;

    int *stop;
    cudaMallocManaged(&stop, sizeof(int));
    cudaMemcpy(stop, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int *share_id;
    cudaMallocManaged(&share_id, sizeof(int));
    cudaMemcpy(share_id, &zero, sizeof(int), cudaMemcpyHostToDevice);

    unsigned char **out_g;
    unsigned char *out_t[MAX_SHARES];

    cudaMalloc((void **)&out_g, MAX_SHARES*sizeof(unsigned char *));

    for (int i = 0; i < MAX_SHARES; ++i) {
        cudaMalloc((void **)&out_t[i], sizeof(unsigned char) * TOTAL_SIZE);
    }
    cudaMemcpy(out_g, out_t, sizeof(unsigned char *) * MAX_SHARES, cudaMemcpyHostToDevice);

    for (int i = 0; i < MAX_SHARES; ++i) {
        cudaMemset(out_t[i], 0, sizeof(unsigned char) * TOTAL_SIZE);
    }

    cudaMemcpyToSymbol(share_chunk_c, share_chunk, sizeof(char) * 64);
    cudaMemcpyToSymbol(share_difficulty_c, &share_difficulty, sizeof(size_t));

    size_t num_threads = threads;
    if (num_threads == 0) {
        num_threads = deviceProp.maxThreadsPerBlock;
    }
    size_t num_blocks = blocks;
    if (num_blocks == 0) {
        num_blocks = (deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor) / num_threads;
    }

    cudaError_t err;
    cudaEvent_t start, end;
    uint loops_count = 0;

    while (*share_id == 0 && loops_count < 20) {
        float elapsed_ms = 0.0f;

        err = cudaEventCreate(&start);
        if (err != cudaSuccess) {
            printf("Failed to create start event: %s\n", cudaGetErrorString(err));
        }

        err = cudaEventCreate(&end);
        if (err != cudaSuccess) {
            printf("Failed to create end event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
        }

        err = cudaEventRecord(start, 0);
        if (err != cudaSuccess) {
            printf("Failed to record start event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        time_t now = time(NULL);

        prefix[98] = now & 0xFF;
        prefix[99] = (now >> 8) & 0xFF;
        prefix[100] = (now >> 16) & 0xFF;
        prefix[101] = (now >> 24) & 0xFF;

        cudaMemcpyToSymbol(prefix_c, prefix, sizeof(unsigned char) * (TOTAL_SIZE-4));

        miner<<<num_blocks,num_threads>>> (out_g, stop, share_id);
        checkCudaErrors(cudaDeviceSynchronize());

        err = cudaEventRecord(end, 0);
        if (err != cudaSuccess) {
            printf("Failed to record end event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        err = cudaEventSynchronize(end);
        if (err != cudaSuccess) {
            printf("Failed to synchronize end event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        err = cudaEventElapsedTime(&elapsed_ms, start, end);
        if (err != cudaSuccess) {
            printf("Failed to get elapsed time: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        if (loops_count % 4 == 0 && !silent) {
            printf("Device: %s\n", deviceProp.name);

            float hashrate = 4294967296.0 / (elapsed_ms / 1000.0) / 1000000000.0;
            printf("Hashrate: %.2f GH/s\n", hashrate);
        }

        if (*share_id > 0) {
            unsigned char *out = (unsigned char *)malloc(sizeof(unsigned char) * TOTAL_SIZE);

            for (int i = 0; i < MIN(*share_id, MAX_SHARES); ++i) {
                cudaMemcpy(out, out_t[i], sizeof(unsigned char) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
                share(poolUrl, bin2hex(out, TOTAL_SIZE), pending_transactions_hashes, pending_transactions_count, id);
            }
            *share_id = 0;
            free(out);
        }

        *stop = 0;
        loops_count++;
    }

    for (int i = 0; i < MAX_SHARES; ++i) {
        cudaFree(out_t[i]);
    }
    cudaFree(out_g);

    cudaFree(stop);
    cudaFree(share_id);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}