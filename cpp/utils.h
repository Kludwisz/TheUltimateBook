#pragma once
#include <cinttypes>
#include <cstdlib>
#include <cstdio>

#define ONLY_TOP_BIT    (0x8000000000000000ULL)
#define FULL_64         (0xffffffffffffffffULL)


#define DEBUG(...) { \
    if (DEBUG_MODE) { \
        printf("%s %d   ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
    } \
}
