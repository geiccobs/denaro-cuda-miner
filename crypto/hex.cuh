//
// Created by User on 24/03/2023.
//

#ifndef C_CUDA_POOL_HEX_CUH
#define C_CUDA_POOL_HEX_CUH

int hexchr2bin(const char hex, char *out);
size_t hexs2bin(const char *hex, unsigned char **out);
char *bin2hex(const unsigned char *bin, size_t len);

#endif //C_CUDA_POOL_HEX_CUH
