#include <stdio.h>
#include <cuda.h>
#include "sha256.cuh"

#define TOTAL_SIZE 108

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

__global__ void miner(unsigned char *hash_prefix, char *share_chunk, size_t share_difficulty, char *charset, unsigned char *out, int *stop, int step) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char _hex[TOTAL_SIZE];
    for (int i = 0; i < TOTAL_SIZE-4; ++i) {
        _hex[i] = hash_prefix[i];
    }

    SHA256_CTX prefix_ctx;
    sha256_init(&prefix_ctx);
    sha256_update(&prefix_ctx, _hex, TOTAL_SIZE-4);

    while (*stop != 1) {
        _hex[TOTAL_SIZE-4] = (index >> 24) & 0xFF;
        _hex[TOTAL_SIZE-3] = (index >> 16) & 0xFF;
        _hex[TOTAL_SIZE-2] = (index >> 8) & 0xFF;
        _hex[TOTAL_SIZE-1] = index & 0xFF;

        SHA256_CTX ctx;
        memcpy(&ctx, &prefix_ctx, sizeof(SHA256_CTX));
        sha256_update(&ctx, _hex + (TOTAL_SIZE-4), 4);

        unsigned char hash[32];
        sha256_final(&ctx, hash);

        char hash_hex[64];
        sha256_to_hex(hash, hash_hex);

        if (compare(hash_hex, share_chunk, share_difficulty)) {
            *stop = 1;

            for (int i = 0; i < TOTAL_SIZE; ++i) {
                out[i] = _hex[i];
            }
            break;
        } else if (index == 0xFFFFFFFF) {
            *stop = 1;
            break;
        }

        index += step;
    }
}

extern "C" {
    void start(const int device_id, const int threads, const int blocks, unsigned char *prefix, char *share_chunk, int share_difficulty, char *charset, unsigned char *out) {
        auto res = cudaSetDevice(device_id);
        if (res != cudaSuccess) {
            printf("Error setting device: %s\n", cudaGetErrorString(res));
            return;
        }

        /*cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        printf("Using device %d: %s\n", device_id, deviceProp.name);*/

        checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));

        int *stop = 0;
        cudaMallocManaged(&stop, sizeof(int));
        cudaMemcpy(stop, 0, sizeof(int), cudaMemcpyHostToDevice);

        char *share_chunk_g;
        cudaMalloc(&share_chunk_g, sizeof(char) * share_difficulty);
        cudaMemcpy(share_chunk_g, share_chunk, sizeof(char) * share_difficulty, cudaMemcpyHostToDevice);

        char *charset_g;
        cudaMalloc(&charset_g, sizeof(char) * 16);
        cudaMemcpy(charset_g, charset, sizeof(char) * 16, cudaMemcpyHostToDevice);

        unsigned char *prefix_g;
        cudaMalloc(&prefix_g, sizeof(unsigned char) * (TOTAL_SIZE-4));
        cudaMemcpy(prefix_g, prefix, sizeof(unsigned char) * (TOTAL_SIZE-4), cudaMemcpyHostToDevice);

        unsigned char *out_g;
        cudaMalloc(&out_g, sizeof(unsigned char) * TOTAL_SIZE);

        miner<<<threads,blocks>>> (prefix_g, share_chunk_g, (size_t)share_difficulty, charset_g, out_g, stop, blocks * threads);

        cudaMemcpy(out, out_g, sizeof(unsigned char) * TOTAL_SIZE, cudaMemcpyDeviceToHost);

        checkCudaErrors(cudaDeviceSynchronize());

        cudaFree(stop);
        cudaFree(charset_g);
        cudaFree(prefix_g);
        cudaFree(out_g);
    }
}