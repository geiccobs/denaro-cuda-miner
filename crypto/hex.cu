#include "hex.cuh"

int hexchr2bin(const char hex, char *out) {
    if (out == NULL)
        return 0;

    if (hex >= '0' && hex <= '9') {
        *out = hex - '0';
    } else if (hex >= 'A' && hex <= 'F') {
        *out = hex - 'A' + 10;
    } else if (hex >= 'a' && hex <= 'f') {
        *out = hex - 'a' + 10;
    } else {
        return 0;
    }
    return 1;
}

size_t hexs2bin(const char *hex, unsigned char **out) {
    size_t len;
    char b1, b2;

    if (hex == NULL || *hex == '\0' || out == NULL)
        return 0;

    len = strlen(hex);
    if (len % 2 != 0)
        return 0;
    len /= 2;

    *out = (unsigned char *) malloc(len);
    memset(*out, 'A', len);
    for (int i = 0; i < len; ++i) {
        if (!hexchr2bin(hex[i*2], &b1) || !hexchr2bin(hex[i*2+1], &b2)) {
            return 0;
        }
        (*out)[i] = (b1 << 4) | b2;
    }
    return len;
}

char *bin2hex(const unsigned char *bin, size_t len) {
    char *out;

    if (bin == NULL || len == 0)
        return NULL;

    out = (char *) malloc(len*2+1);
    for (int i = 0; i < len; ++i) {
        out[i*2]   = "0123456789abcdef"[bin[i] >> 4];
        out[i*2+1] = "0123456789abcdef"[bin[i] & 0x0F];
    }
    out[len*2] = '\0';

    return out;
}