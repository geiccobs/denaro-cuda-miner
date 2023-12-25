#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#include "crypto/base58.cuh"
#include "crypto/sha256.cuh"
#include "requests.cuh"
#include "kernel/kernel.cuh"
#include "crypto/hex.cuh"

#define MAX_ADDRESSES 64

GpuSettings gpuSettings = {0};
LocalSettings localSettings = {0};

ManagerData managerData = {0};

void setDefaultSettings() {
    localSettings.address = (char **) malloc(sizeof(char *) * MAX_ADDRESSES);
    for (int i = 0; i < MAX_ADDRESSES; ++i) {
        localSettings.address[i] = (char *) malloc(sizeof(char) * 45);
        strcpy(localSettings.address[i], "\0");
    }
    localSettings.devFee = 5;
    localSettings.loops = 0;

    gpuSettings.nodeUrl = (char *) malloc(sizeof(char) * 128);
    strcpy(gpuSettings.nodeUrl, "https://denaro-node.gaetano.eu.org/");
    gpuSettings.poolUrl = (char *) malloc(sizeof(char) * 128);
    strcpy(gpuSettings.poolUrl, "https://denaro-pool.gaetano.eu.org/");
    gpuSettings.silent = false;
    gpuSettings.verbose = false;
    gpuSettings.deviceId = 0;
    gpuSettings.threads = 0;
    gpuSettings.blocks = 0;
    gpuSettings.shareDifficulty = 9;
}

void parseArguments(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            printf("Options:\n");
            printf("  --help\t\tShow this help\n");
            printf("  --address\t\tSet your denaro address (https://t.me/DenaroCoinBot)\n");
            printf("  --node\t\tSet the node url\n");
            printf("  --pool\t\tSet the pool url\n");
            printf("  --silent\t\tSilent mode (no output)\n");
            printf("  --verbose\t\tVerbose mode (debug output)\n");
            printf("  --device\t\tSet the gpu device id\n");
            printf("  --threads\t\tSet the gpu threads, 0 for auto\n");
            printf("  --blocks\t\tSet the gpu blocks, 0 for auto\n");
            printf("  --share\t\tSet the share difficulty\n");
            printf("  --fee\t\t\tSet the dev fee (1 every X blocks are mined by the dev)\n");

            exit(0);
        } else if (strcmp(argv[i], "--address") == 0) {
            if (i + 1 < argc) {
                char *token = strtok(argv[i + 1], ",");
                int j = 0;
                while (token != NULL) {
                    strcpy(localSettings.address[j], token);
                    token = strtok(NULL, ",");
                    j++;
                }
            }
        } else if (strcmp(argv[i], "--node") == 0) {
            if (i + 1 < argc) {
                strcpy(gpuSettings.nodeUrl, argv[i + 1]);
            }
        } else if (strcmp(argv[i], "--pool") == 0) {
            if (i + 1 < argc) {
                strcpy(gpuSettings.poolUrl, argv[i + 1]);
            }
        } else if (strcmp(argv[i], "--silent") == 0) {
            gpuSettings.silent = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            gpuSettings.verbose = true;
        } else if (strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                gpuSettings.deviceId = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                gpuSettings.threads = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--blocks") == 0) {
            if (i + 1 < argc) {
                gpuSettings.blocks = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--share") == 0) {
            if (i + 1 < argc) {
                gpuSettings.shareDifficulty = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--fee") == 0) {
            if (i + 1 < argc) {
                localSettings.devFee = strtol(argv[i + 1], NULL, 10);
            }
        }
    }

    if (strlen(localSettings.address[0]) == 0) {
        printf("Please specify your denaro address (https://t.me/DenaroCoinBot): ");
        scanf("%s", localSettings.address[0]);
    }
}

char *get_transactions_merkle_tree(char transactions[512][64 + 1], size_t transactions_count) {
    unsigned char *full_data = (unsigned char *) malloc(64 * transactions_count);
    unsigned char *data;

    uint index = 0;
    for (int i = 0; i < transactions_count; ++i) {
        size_t len = hexs2bin(transactions[i], &data);
        for (int x = 0; x < len; ++x) {
            full_data[index++] = data[x];
        }
    }

    unsigned char *hash = (unsigned char *) malloc(32);
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, full_data, index);
    sha256_final(&ctx, hash);

    return bin2hex(hash, 32);
}

char *get_random_address() {
    int random = rand() % MAX_ADDRESSES;
    while (strlen(localSettings.address[random]) == 0) {
        random = rand() % MAX_ADDRESSES;
    }
    return localSettings.address[random];
}

void generate_prefix() {
    uint difficulty = (uint) managerData.miningInfo.result.difficulty;

    char *chunk = (char *) malloc((64 + 1) * sizeof(char));
    size_t hash_len = strlen(managerData.miningInfo.result.last_block.hash);
    for (int i = hash_len - difficulty; i < hash_len; ++i) {
        chunk[i - (hash_len - difficulty)] = managerData.miningInfo.result.last_block.hash[i];
    }

    if (gpuSettings.shareDifficulty > difficulty) {
        gpuSettings.shareDifficulty = difficulty;
    }
    for (int i = 0; i < gpuSettings.shareDifficulty; ++i) {
        managerData.shareChunk[i] = chunk[i];
    }

    size_t address_bytes_len = 33;
    unsigned char address_bytes[address_bytes_len];

    if (localSettings.devFee != 0 && localSettings.loops != 0 && localSettings.loops % localSettings.devFee == 0) {
        const char *mining_address_dev = "DnAmhfPcckW4yDCVdaMQtPs6CkSfsQNyDrJ6kZzanJpty\0";
        b58tobin(address_bytes, &address_bytes_len, mining_address_dev, strlen(mining_address_dev));
    } else {
        b58tobin(address_bytes, &address_bytes_len, managerData.miningAddress, strlen(managerData.miningAddress));
    }

    char *transactions_merkle_tree = get_transactions_merkle_tree(managerData.miningInfo.result.pending_transactions_hashes, managerData.miningInfo.result.pending_transactions_count);

    // version, not supporting v1
    managerData.prefix[0] = 2;

    // previous block hash
    unsigned char *previous_block_hash;
    size_t previous_block_hash_len = hexs2bin(managerData.miningInfo.result.last_block.hash, &previous_block_hash);

    for (int i = 0; i < previous_block_hash_len; ++i) {
        managerData.prefix[i + 1] = previous_block_hash[i];
    }

    // address bytes
    for (int i = 0; i < address_bytes_len; ++i) {
        managerData.prefix[i + 33] = address_bytes[i];
    }

    // transactions merkle tree
    unsigned char *transactions_merkle_tree_bytes;
    size_t transactions_merkle_tree_bytes_len = hexs2bin(transactions_merkle_tree, &transactions_merkle_tree_bytes);

    for (int i = 0; i < transactions_merkle_tree_bytes_len; ++i) {
        managerData.prefix[i + 33 + 33] = transactions_merkle_tree_bytes[i];
    }

    // difficulty bytes (which is 2 bytes, difficulty * 10) on bytes 99 and 100
    unsigned char *difficulty_bytes = (unsigned char *) malloc(2 * sizeof(unsigned char));
    uint difficulty_10 = difficulty * 10;
    memcpy(difficulty_bytes, &difficulty_10, sizeof(unsigned char) * 2);

    managerData.prefix[102] = difficulty_bytes[0];
    managerData.prefix[103] = difficulty_bytes[1];

    free(chunk);
    free(transactions_merkle_tree);
    free(previous_block_hash);
    free(transactions_merkle_tree_bytes);
    free(difficulty_bytes);
}

void manager_init() {
    if (managerData.stop != NULL) free(managerData.stop);
    managerData.stop = (bool *) malloc(sizeof(bool));
    *managerData.stop = false;

    managerData.shares = 0;

    if (managerData.shareChunk != NULL) free(managerData.shareChunk);
    managerData.shareChunk = (char *) malloc((64 + 1) * sizeof(char));
}

bool manager_load() {
    managerData.miningInfo = get_mining_info(gpuSettings.nodeUrl);
    if (!managerData.miningInfo.ok) {
        if (!gpuSettings.silent) {
            fprintf(stderr, "Failed to get mining info\n");
        }
        return false;
    }

    managerData.miningAddress = get_mining_address(gpuSettings.poolUrl, get_random_address());
    if (managerData.miningAddress == NULL) {
        if (!gpuSettings.silent) {
            fprintf(stderr, "Failed to get mining address\n");
        }
        return false;
    }
    return true;
}

void *manager(void *arg) {
    MiningInfo mining_info;
    while (true) {
        mining_info = get_mining_info(gpuSettings.nodeUrl);
        if (mining_info.ok && !(*managerData.stop) && mining_info.result.last_block.id != managerData.miningInfo.result.last_block.id) {
            *managerData.stop = true;
        }
        sleep(1);
    }
}

int main(int argc, char *argv[]) {
    setDefaultSettings();
    parseArguments(argc, argv);

    srand(time(NULL));

    manager_init();

    pthread_t manager_thread;
    pthread_create(&manager_thread, NULL, manager, NULL);

    while (true) {
        if (!manager_load()) {
            sleep(1);
            continue;
        }
        generate_prefix();

        start(&gpuSettings, &managerData);

        manager_init();
        localSettings.loops++;
    }
    return 0;
}