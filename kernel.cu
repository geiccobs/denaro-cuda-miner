#include <stdio.h>
#include <cuda.h>
#include "sha256.cuh"

#define TOTAL_SIZE 108
#define MAX_SHARES 16

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__ void sha256_to_hex(unsigned char* data, char pout[64]) {
    const char* hex = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        pout[i * 2] = hex[data[i] >> 4];
        pout[i * 2 + 1] = hex[data[i] & 0x0f];
    }
}

__device__ bool compare(const char* str_a, const char* str_b, unsigned len) {
    for (int i = 0; i < len; ++i) {
        if (str_a[i] != str_b[i])
            return false;
    }
    return true;
}

__global__ void miner(unsigned char *hash_prefix, char *share_chunk, size_t share_difficulty, unsigned char **out, int *stop, int *share_id) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char _hex[TOTAL_SIZE];
    for (int i = 0; i < TOTAL_SIZE-4; ++i) {
        _hex[i] = hash_prefix[i];
    }

    SHA256_CTX prefix_ctx;
    sha256_init(&prefix_ctx);
    sha256_update(&prefix_ctx, _hex, TOTAL_SIZE-4);

    while (*stop != 1) {
        _hex[TOTAL_SIZE-4] = index >> 24;
        _hex[TOTAL_SIZE-3] = index >> 16;
        _hex[TOTAL_SIZE-2] = index >> 8;
        _hex[TOTAL_SIZE-1] = index;

        SHA256_CTX ctx;
        memcpy(&ctx, &prefix_ctx, sizeof(SHA256_CTX));
        sha256_update(&ctx, _hex + (TOTAL_SIZE-4), 4);

        unsigned char hash[32];
        sha256_final(&ctx, hash);

        char hash_hex[64];
        sha256_to_hex(hash, hash_hex);

        if (compare(hash_hex, share_chunk, share_difficulty)) {
            memcpy(out[*share_id], _hex, sizeof(unsigned char) * TOTAL_SIZE);
            *share_id += 1;
        }
        if (index == 0xFFFFFFFF || *share_id == MAX_SHARES) {
            *stop = 1;
        }
        index += blockDim.x * gridDim.x;
    }
}

extern "C" {
    void start(const int device_id, const int threads, const int blocks, unsigned char *prefix, char *share_chunk, int share_difficulty, char *device_name, float *hashrate, unsigned char **out) {
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

        char *share_chunk_g;
        cudaMalloc(&share_chunk_g, sizeof(char) * share_difficulty);
        cudaMemcpy(share_chunk_g, share_chunk, sizeof(char) * share_difficulty, cudaMemcpyHostToDevice);

        unsigned char *prefix_g;
        cudaMalloc(&prefix_g, sizeof(unsigned char) * (TOTAL_SIZE-4));
        cudaMemcpy(prefix_g, prefix, sizeof(unsigned char) * (TOTAL_SIZE-4), cudaMemcpyHostToDevice);

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

        miner<<<threads,blocks>>> (prefix_g, share_chunk_g, share_difficulty, out_g, stop, share_id);
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

        *hashrate = 4294967296.0 / (elapsed_ms / 1000.0) / 1000000.0;

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
        cudaFree(share_chunk_g);
        cudaFree(prefix_g);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
}