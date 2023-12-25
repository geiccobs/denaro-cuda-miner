#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdbool.h>

#include "kernel.cuh"
#include "sha256.cuh"
#include "../requests.cuh"
#include "../crypto/hex.cuh"

#define TOTAL_SIZE 108
#define MAX_SHARES 16

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define CLEAR() printf("\033[H\033[J")

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

__global__ void miner(unsigned char **out, bool *stop, unsigned char *prefix, int *share_id) {
    const __restrict__ uint32_t tid = threadIdx.x;

    __shared__ SHA256_CTX prefix_ctx;
    if (tid == 0) {
        sha256_init_dev(&prefix_ctx);
        sha256_update_dev(&prefix_ctx, prefix, sizeof(unsigned char) * (TOTAL_SIZE-4));
    }
    __syncthreads();

    unsigned char _hex[TOTAL_SIZE];
    memcpy(_hex, prefix, sizeof(unsigned char) * (TOTAL_SIZE-4));

    SHA256_CTX ctx;
    unsigned char hash[32];
    char hash_hex[64];

    for (uint32_t index = blockIdx.x * blockDim.x + tid; !(*stop); index += blockDim.x * gridDim.x) {
        _hex[TOTAL_SIZE-1] = index;
        _hex[TOTAL_SIZE-2] = index >> 8;
        _hex[TOTAL_SIZE-3] = index >> 16;
        _hex[TOTAL_SIZE-4] = index >> 24;

        memcpy(&ctx, &prefix_ctx, sizeof(SHA256_CTX));

        sha256_update_dev(&ctx, _hex + (TOTAL_SIZE-4), sizeof(unsigned char) * 4);
        sha256_final_dev(&ctx, hash);
        sha256_to_hex(hash, hash_hex);

        if (is_valid(hash_hex)) {
            int id = atomicAdd(share_id, 1);
            memcpy(out[id], _hex, sizeof(unsigned char) * TOTAL_SIZE);

            if (id >= MAX_SHARES-2) {
                *stop = true;
            }
        }

        if (index >= 0xFFFFFFFF) {
            *stop = true;
        }
    }
}

void start(GpuSettings *settings, ManagerData *managerData) {
    auto res = cudaSetDevice(settings->deviceId);
    if (res != cudaSuccess) {
        printf("Error setting device: %s\n", cudaGetErrorString(res));
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, settings->deviceId);

    checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));

    // allocate memory on the device
    int zero = 0;

    bool *stop_g;
    cudaMallocManaged(&stop_g, sizeof(bool));
    cudaMemcpy(stop_g, &zero, sizeof(bool), cudaMemcpyHostToDevice);

    int *share_id;
    cudaMallocManaged(&share_id, sizeof(int));
    cudaMemcpy(share_id, &zero, sizeof(int), cudaMemcpyHostToDevice);

    unsigned char *prefix_g;
    cudaMallocManaged(&prefix_g, sizeof(unsigned char) * (TOTAL_SIZE-4));

    unsigned char **out_g;
    cudaMallocManaged(&out_g, sizeof(unsigned char*) * MAX_SHARES);

    for (int i = 0; i < MAX_SHARES; ++i) {
        cudaMallocManaged(&out_g[i], sizeof(unsigned char) * TOTAL_SIZE);
        cudaMemset(out_g[i], 0, sizeof(unsigned char) * TOTAL_SIZE);
    }

    cudaMemcpyToSymbol(share_chunk_c, managerData->shareChunk, sizeof(char) * 64);
    cudaMemcpyToSymbol(share_difficulty_c, &settings->shareDifficulty, sizeof(size_t));

    size_t num_threads = settings->threads;
    if (num_threads == 0) {
        num_threads = deviceProp.maxThreadsPerBlock;
    }
    size_t num_blocks = settings->blocks;
    if (num_blocks == 0) {
        num_blocks = (deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor) / num_threads;
    }

    if (settings->verbose) printf("Starting miner with %zu blocks and %zu threads\n", num_blocks, num_threads);

    cudaError_t err;
    cudaEvent_t start;
    cudaEvent_t end;
    uint loops_count = 0;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        printf("Failed to create start event: %s\n", cudaGetErrorString(err));
    }

    err = cudaEventCreate(&end);
    if (err != cudaSuccess) {
        printf("Failed to create end event: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
    }

    while (!(*managerData->stop)) {
        float elapsed_ms = 0.0f;

        err = cudaEventRecord(start, 0);
        if (err != cudaSuccess) {
            printf("Failed to record start event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

        time_t now = time(NULL);

        cudaMemcpy(prefix_g, managerData->prefix, sizeof(unsigned char) * (TOTAL_SIZE-4), cudaMemcpyHostToDevice);

        prefix_g[98] = now & 0xFF;
        prefix_g[99] = (now >> 8) & 0xFF;
        prefix_g[100] = (now >> 16) & 0xFF;
        prefix_g[101] = (now >> 24) & 0xFF;

        miner<<<num_blocks,num_threads>>> (out_g, stop_g, prefix_g, share_id);
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

        if (!settings->silent) {
            float hashrate = (pow(2, 32) - 1) / (elapsed_ms / 1000.0) / pow(10, 9);

            CLEAR();
            printf("Denaro GPU Miner\n\n");
            printf("Device: %s\n", deviceProp.name);
            printf("Threads: %zu\n", num_threads);
            printf("Blocks: %zu\n\n", num_blocks);

            printf("Node: %s\n", settings->nodeUrl);
            printf("Pool: %s\n\n", settings->poolUrl);

            printf("Accepted shares: %d\n\n", managerData->shares);

            printf("Hashrate: %.2f GH/s\n", hashrate);
        }

        if (*share_id > 0) {
            Share resp;

            unsigned char *out;
            cudaMallocManaged(&out, sizeof(unsigned char) * TOTAL_SIZE);

            for (int i = 0; i < MIN(*share_id, MAX_SHARES); ++i) {
                cudaMemcpy(out, out_g[i], sizeof(unsigned char) * TOTAL_SIZE, cudaMemcpyDeviceToHost);

                if (out[0] == 2) {
                    resp = share(
                            settings->poolUrl,
                            bin2hex(out, TOTAL_SIZE),
                            managerData->miningInfo.result.pending_transactions_hashes,
                            managerData->miningInfo.result.pending_transactions_count,
                            managerData->miningInfo.result.last_block.id + 1
                    );
                    if (resp.ok) {
                        if (settings->verbose) {
                            printf("Share accepted: %s\n", bin2hex(out, TOTAL_SIZE));
                        }
                        managerData->shares++;
                    } else {
                        if (settings->verbose) {
                            printf("Share not accepted: %s\n", resp.error);
                        }
                        *managerData->stop = true;
                    }
                }
                cudaMemset(out_g[i], 0, sizeof(unsigned char) * TOTAL_SIZE);
            }
            *share_id = 0;
        }

        *stop_g = false;
        loops_count++;
    }

    for (int i = 0; i < MAX_SHARES; ++i) {
        cudaFree(out_g[i]);
    }
    cudaFree(out_g);

    cudaFree(stop_g);
    cudaFree(share_id);
    cudaFree(prefix_g);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}