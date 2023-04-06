#ifndef C_CUDA_POOL_KERNEL_CUH
#define C_CUDA_POOL_KERNEL_CUH

#include "../requests.cuh"

#define TOTAL_SIZE 108

typedef struct {
    char **address; // denaro address (https://t.me/DenaroCoinBot)
    uint devFee; // dev fee (1 every X blocks are mined using the dev address)
    uint loops; // loops the main function has done (used for dev fee) - increases on "share not accepted" response
} LocalSettings;

typedef struct {
    char *nodeUrl;  // denaro node url
    char *poolUrl; // denaro pool url
    bool silent; // silent mode (no output)
    bool verbose; // verbose mode (debug output)
    uint deviceId; // gpu device id
    uint threads; // gpu threads - 0 for auto
    uint blocks; // gpu blocks - 0 for auto
    uint shareDifficulty; // share difficulty
} GpuSettings;

typedef struct {
    bool *stop;
    char *miningAddress;
    char *shareChunk;
    uint shares;
    unsigned char prefix[104];
    MiningInfo miningInfo;
} ManagerData;

void start(GpuSettings *settings, ManagerData *managerData);

#endif //C_CUDA_POOL_KERNEL_CUH
