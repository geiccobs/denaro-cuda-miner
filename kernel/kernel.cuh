//
// Created by User on 24/03/2023.
//

#ifndef C_CUDA_POOL_KERNEL_CUH
#define C_CUDA_POOL_KERNEL_CUH

#define TOTAL_SIZE 108
void start(const int device_id, const int threads, const int blocks, unsigned char prefix[TOTAL_SIZE-4], char *share_chunk, size_t share_difficulty, unsigned char **out, bool silent);

#endif //C_CUDA_POOL_KERNEL_CUH
