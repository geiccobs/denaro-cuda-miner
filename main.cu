#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <threads.h>

#include "crypto/base58.cuh"
#include "crypto/sha256.cuh"
#include "requests.cuh"
#include "kernel/kernel.cuh"
#include "crypto/hex.cuh"

struct Settings {
    char address[128][45 + 1]; // denaro address (https://t.me/DenaroCoinBot)
    char nodeUrl[2048 + 1];  // denaro node url
    char poolUrl[2048 + 1]; // denaro pool url
    bool silent; // silent mode (no output)
    bool verbose; // verbose mode (debug output)
    uint deviceId; // gpu device id
    uint threads; // gpu threads - 0 for auto
    uint blocks; // gpu blocks - 0 for auto
    uint shareDifficulty; // share difficulty
    uint devFee; // dev fee (1 every X shares are sent to the dev)
} settings;

struct ManagerData {
    unsigned char **out;
    MiningInfo miningInfo;
} managerData;

void setDefaultSettings()
{
    for (int i = 0; i < 128; ++i) {
        strcpy(settings.address[i], "");
    }
    strcpy(settings.nodeUrl, "https://denaro-node.gaetano.eu.org/");
    strcpy(settings.poolUrl, "https://denaro-pool.gaetano.eu.org/");
    settings.silent = false;
    settings.verbose = false;
    settings.deviceId = 0;
    settings.threads = 0;
    settings.blocks = 0;
    settings.shareDifficulty = 9;
    settings.devFee = 5;
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
            printf("  --fee\t\t\tSet the dev fee (1 every X shares are sent to the dev)\n");
        } else if (strcmp(argv[i], "--address") == 0) {
            if (i + 1 < argc) {
                char *token = strtok(argv[i + 1], ",");
                int j = 0;
                while (token != NULL) {
                    strcpy(settings.address[j], token);
                    token = strtok(NULL, ",");
                    j++;
                }
            }
        } else if (strcmp(argv[i], "--node") == 0) {
            if (i + 1 < argc) {
                strcpy(settings.nodeUrl, argv[i + 1]);
            }
        } else if (strcmp(argv[i], "--pool") == 0) {
            if (i + 1 < argc) {
                strcpy(settings.poolUrl, argv[i + 1]);
            }
        } else if (strcmp(argv[i], "--silent") == 0) {
            settings.silent = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            settings.verbose = true;
        } else if (strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                settings.deviceId = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                settings.threads = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--blocks") == 0) {
            if (i + 1 < argc) {
                settings.blocks = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--share") == 0) {
            if (i + 1 < argc) {
                settings.shareDifficulty = strtol(argv[i + 1], NULL, 10);
            }
        } else if (strcmp(argv[i], "--fee") == 0) {
            if (i + 1 < argc) {
                settings.devFee = strtol(argv[i + 1], NULL, 10);
            }
        }
    }

    if (strlen(settings.address[0]) == 0) {
        printf("Please specify your denaro address (https://t.me/DenaroCoinBot): ");
        scanf("%s", settings.address[0]);
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
    int random = rand() % 128;
    while (strlen(settings.address[random]) == 0) {
        random = rand() % 128;
    }
    return settings.address[random];
}

void miner(const char *mining_address) {
    uint difficulty = managerData.miningInfo.result.difficulty;
    uint idifficulty = (uint) difficulty;

    char *chunk = (char *) malloc((64 + 1) * sizeof(char));
    size_t hash_len = strlen(managerData.miningInfo.result.last_block.hash);
    for (int i = hash_len - idifficulty; i < hash_len; ++i) {
        chunk[i - (hash_len - idifficulty)] = managerData.miningInfo.result.last_block.hash[i];
    }

    char *share_chunk = (char *) malloc((64 + 1) * sizeof(char));
    if (settings.shareDifficulty > idifficulty) {
        settings.shareDifficulty = idifficulty;
    }
    for (int i = 0; i < settings.shareDifficulty; ++i) {
        share_chunk[i] = chunk[i];
    }

    unsigned char address_bytes[32 + 1];
    size_t address_bytes_len = 33;

    // TODO: devfee
    b58tobin(address_bytes, &address_bytes_len, mining_address, strlen(mining_address));

    char *transactions_merkle_tree = get_transactions_merkle_tree(managerData.miningInfo.result.pending_transactions_hashes, managerData.miningInfo.result.pending_transactions_count);

    unsigned char prefix[104];

    // version, not supporting v1
    prefix[0] = 2;

    // previous block hash
    unsigned char *previous_block_hash;
    size_t previous_block_hash_len = hexs2bin(managerData.miningInfo.result.last_block.hash, &previous_block_hash);

    for (int i = 0; i < previous_block_hash_len; ++i) {
        prefix[i + 1] = previous_block_hash[i];
    }

    // address bytes
    for (int i = 0; i < address_bytes_len; ++i) {
        prefix[i + 33] = address_bytes[i];
    }

    // transactions merkle tree
    unsigned char *transactions_merkle_tree_bytes;
    size_t transactions_merkle_tree_bytes_len = hexs2bin(transactions_merkle_tree, &transactions_merkle_tree_bytes);

    for (int i = 0; i < transactions_merkle_tree_bytes_len; ++i) {
        prefix[i + 33 + 33] = transactions_merkle_tree_bytes[i];
    }

    // difficulty bytes (which is 2 bytes, idifficulty * 10) on bytes 99 and 100
    unsigned char *difficulty_bytes = (unsigned char *) malloc(2 * sizeof(unsigned char));
    uint idifficulty_10 = idifficulty * 10;
    memcpy(difficulty_bytes, &idifficulty_10, sizeof(unsigned char) * 2);

    prefix[102] = difficulty_bytes[0];
    prefix[103] = difficulty_bytes[1];

    start(
            settings.deviceId,
            settings.threads,
            settings.blocks,
            prefix,
            share_chunk,
            settings.shareDifficulty,
            managerData.out,
            settings.silent
    );

    free(chunk);
    free(share_chunk);
    free(transactions_merkle_tree);
    free(previous_block_hash);
    free(transactions_merkle_tree_bytes);
    free(difficulty_bytes);
}

int manager(void *arg) {
    Share resp;

    while (true) {
        for (int i = 0; i < 16; ++i) {
            if (managerData.out[i][0] == 2) {
                resp = share(settings.poolUrl, bin2hex(managerData.out[i], TOTAL_SIZE), managerData.miningInfo.result.pending_transactions_hashes, managerData.miningInfo.result.pending_transactions_count, managerData.miningInfo.result.last_block.id+1);
                if (resp.ok) {
                    printf("Share accepted\n");
                } else {
                    printf("Share not accepted: %s\n", resp.error);
                }
                memset(managerData.out[i], 0, 108);
            }
        }
        sleep(1);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    setDefaultSettings();
    parseArguments(argc, argv);

    srand(time(NULL));

    managerData.out = (unsigned char **) malloc(16 * sizeof(unsigned char *));
    for (int i = 0; i < 16; ++i) {
        managerData.out[i] = (unsigned char *) malloc(108 * sizeof(unsigned char));
        memset(managerData.out[i], 0, 108);
    }

    thrd_t manager_thread;
    thrd_create(&manager_thread, manager, NULL);

    while (true) {
        const char *mining_address = get_mining_address(settings.poolUrl, get_random_address());
        if (mining_address == NULL) {
            fprintf(stderr, "Failed to get mining address\n");
            sleep(1);
            continue;
        }

        managerData.miningInfo = get_mining_info(settings.nodeUrl);
        if (!managerData.miningInfo.ok) {
            fprintf(stderr, "Failed to get mining info\n");
            sleep(1);
            continue;
        }

        miner(mining_address);
        printf("completed loop\n");
    }

    free(managerData.out);

    return 0;
}