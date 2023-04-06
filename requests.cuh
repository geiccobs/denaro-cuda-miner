#ifndef C_CUDA_POOL_REQUESTS_CUH
#define C_CUDA_POOL_REQUESTS_CUH

#include <stdbool.h>

typedef struct {
    bool ok;
    struct result {
        double difficulty;
        struct last_block {
            uint id;
            char hash[64 + 1];
            char address[45 + 1];
            uint random;
            double difficulty;
            double reward;
            time_t timestamp;
            char content[2048 + 1];
        } last_block;
        char pending_transactions_hashes[512][64 + 1];
        size_t pending_transactions_count;
        char merkle_root[64 + 1];
    } result;
} MiningInfo;

typedef struct {
    bool ok;
    char error[128 + 1];
} Share;

Share share(const char *poolUrl, const char *hash, const char pending_transactions_hashes[512][64 + 1], size_t pending_transactions_count, uint id);
MiningInfo get_mining_info(const char *nodeUrl);
char *get_mining_address(const char *poolUrl, const char *address);

#endif //C_CUDA_POOL_REQUESTS_CUH
