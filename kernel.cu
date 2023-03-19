#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "sha256.cuh"

#define TOTAL_SIZE 108
#define MAX_SHARES 16

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__ __constant__ uint32_t prefix_c[TOTAL_SIZE/4];
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
        sha256_init(&prefix_ctx);
        sha256_update(&prefix_ctx, (unsigned char*)prefix_c, sizeof(uint32_t) * (TOTAL_SIZE-4)/4);
    }
    __syncthreads();

    uint32_t _hex[TOTAL_SIZE/4];
    memcpy(_hex, prefix_c, sizeof(uint32_t) * (TOTAL_SIZE-4)/4);

    SHA256_CTX ctx;
    unsigned char hash[32];
    char hash_hex[64];

    for (uint32_t index = blockIdx.x * blockDim.x + tid; *stop != 1; index += blockDim.x * gridDim.x) {
        _hex[TOTAL_SIZE/4-1] = index;

        memcpy(&ctx, &prefix_ctx, sizeof(SHA256_CTX));
        sha256_update(&ctx, (unsigned char*)&_hex[TOTAL_SIZE/4-1], 4);
        sha256_final(&ctx, hash);
        sha256_to_hex(hash, hash_hex);

        if (is_valid(hash_hex)) {
            int id = atomicAdd(share_id, 1);
            memcpy(out[id], _hex, sizeof(uint32_t) * TOTAL_SIZE/4);

            if (id >= MAX_SHARES-2) {
                *stop = 1;
            }
        }

        if (index >= 0xFFFFFFFF) {
            *stop = 1;
        }
    }
}

extern "C" {
    void start(const int device_id, const int threads, const int blocks, uint32_t *prefix, size_t difficulty, char *share_chunk, size_t share_difficulty, char *device_name, float *hashrate, unsigned char **out) {
        auto res = cudaSetDevice(device_id);
        if (res != cudaSuccess) {
            printf("Error setting device: %s\n", cudaGetErrorString(res));
            return;
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        strcpy(device_name, deviceProp.name);

        checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));

        // allocate memory on the device
        int *stop;
        cudaMallocManaged(&stop, sizeof(int));
        cudaMemcpy(stop, 0, sizeof(int), cudaMemcpyHostToDevice);

        int *share_id;
        cudaMallocManaged(&share_id, sizeof(int));
        cudaMemcpy(share_id, 0, sizeof(int), cudaMemcpyHostToDevice);

        unsigned char **out_g;
        unsigned char *out_t[MAX_SHARES];

        cudaMalloc((void **)&out_g, MAX_SHARES*sizeof(unsigned char *));

        for (int i = 0; i < MAX_SHARES; ++i) {
            cudaMalloc((void **)&out_t[i], sizeof(unsigned char) * TOTAL_SIZE);
        }
        cudaMemcpy(out_g, out_t, sizeof(unsigned char *) * MAX_SHARES, cudaMemcpyHostToDevice);

        for (int i = 0; i < MAX_SHARES; ++i) {
            cudaMemcpy(out_t[i], out[i], sizeof(unsigned char) * TOTAL_SIZE, cudaMemcpyHostToDevice);
        }

        cudaError_t err;
        cudaEvent_t start, end;
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

        prefix[TOTAL_SIZE/4-2] = (prefix[TOTAL_SIZE/4-2] & 0xFFFF) | (difficulty << 16);

        cudaMemcpyToSymbol(share_chunk_c, share_chunk, sizeof(char) * 64);
        cudaMemcpyToSymbol(share_difficulty_c, &share_difficulty, sizeof(size_t));

        uint loops_count = 0;
        while (*share_id == 0 && loops_count < 5) {
            time_t now = time(NULL);
            prefix[TOTAL_SIZE/4-3] = (prefix[TOTAL_SIZE/4-3] & 0xFFFF) | ((now & 0xFFFF) << 16);
            prefix[TOTAL_SIZE/4-2] = (prefix[TOTAL_SIZE/4-2] & 0xFFFF0000) | ((now & 0xFFFF0000) >> 16);

            cudaMemcpyToSymbol(prefix_c, prefix, sizeof(uint32_t) * ((TOTAL_SIZE-4)/4));

            miner<<<threads,blocks>>> (out_g, stop, share_id);
            checkCudaErrors(cudaDeviceSynchronize());

            *stop = 0;
            loops_count++;
        }

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

        *hashrate = (4294967296.0 / (elapsed_ms / 1000.0) / 1000000000.0) * loops_count;

        if (*share_id > 0) {
            for (int i = 0; i < MIN(*share_id, MAX_SHARES); ++i) {
                cudaMemcpy(out[i], out_t[i], sizeof(unsigned char) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
            }
        }

        for (int i = 0; i < MAX_SHARES; ++i) {
            cudaFree(out_t[i]);
        }
        cudaFree(out_g);

        cudaFree(stop);
        cudaFree(share_id);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
}