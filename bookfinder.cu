#ifndef USE_CUDA
#define __constant__
#define __managed__
#define __shared__
#define __host__
#define __device__
#define __global__
#endif

#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include "xoroshiro.h"


// --------------------------------------------------------------------------------
// GPU side code: just go brrrrrr and crunch as many Xoroshiro states as possible.

// kernel input data
constexpr std::uint32_t N_ITER_MASKS = 64 * 2;
constexpr std::uint32_t N_BLOCK_MASKS = 5 * 4 * 2;
constexpr std::uint32_t N_THREAD_MASKS = 2 * 16 * 2;
__device__   std::uint64_t THREAD_MASKS[N_THREAD_MASKS];
__device__   std::uint64_t BLOCK_MASKS[N_BLOCK_MASKS];
__constant__ std::uint64_t ITER_MASKS[N_ITER_MASKS];

// kernel output data
constexpr std::uint32_t N_RESULTS_MAX = 1024;
__managed__ Xoroshiro results[N_RESULTS_MAX];
__managed__ uint32_t result_count = 0;

// the kernel
__global__ void gaussianElimIterKernel(Xoroshiro initialState) {
    // per-thread computations
    std::uint32_t tid; // = threadIdx.x;
    initialState.lo ^= THREAD_MASKS[(tid & 0b1111) * 2];
    initialState.hi ^= THREAD_MASKS[(tid & 0b1111) * 2 + 1];
    tid >>= 4;
    initialState.lo ^= THREAD_MASKS[32 + (tid & 0b1111) * 2];
    initialState.hi ^= THREAD_MASKS[32 + (tid & 0b1111) * 2 + 1];

    // per-block computations
    std::uint32_t bid; // = blockIdx.x;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        initialState.lo ^= BLOCK_MASKS[i*8 + (bid & 0b11) * 2];
        initialState.hi ^= BLOCK_MASKS[i*8 + (bid & 0b11) * 2 + 1];
        bid >>= 2;
    }

    // We have the initial state. Now all remaining state jump operations will be
    // handled using fully precomputed mask values stored in __constant__ memory. 

    for (int iterIndex = 0; iterIndex < 64; iterIndex++) {
        Xoroshiro state = initialState;
        state.lo ^= ITER_MASKS[iterIndex*2];
        state.hi ^= ITER_MASKS[iterIndex*2 + 1];

        if (xNextIntFast(&state, 50) != 0) continue; // 7
        if (xNextIntFast(&state, 50) != 0) continue; // 8
        if (xNextIntFast(&state, 50) != 0) continue; // 9
        if (xNextIntFast(&state, 50) != 0) continue; // 10
        if (xNextIntFast(&state, 50) != 0) continue; // 11

        {// got a partial result
            Xoroshiro out = initialState;
            out.lo ^= ITER_MASKS[iterIndex*2];
            out.hi ^= ITER_MASKS[iterIndex*2 + 1];

            int idx;// = atomicAdd(&result_count, 1);
            results[idx] = out;
        }
    }
}


// -------------------------------------------------------------------------------
// Host side code: parse input args, solve the system of equations, load the 
// correct data into GPU constant memory, print the results.

void setup_search(std::uint64_t eqn_seed) {
    // TODO gen solution, gen masks, copy data to device & constant mem
}

void run(std::uint64_t eqn_seed) {
    setup_search(eqn_seed);

    result_count = 0;
    // TODO run the kernel, synchronize.
}

int main(int argc, char **argv) {
    std::uint64_t start_eqn_seed = 0;
    std::uint32_t num_runs = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0) {
            if (i == argc-1 || sscanf(argv[i+1], "%" PRIu64, &start_eqn_seed) != 1) {
                std::fprintf(stderr, "Illegal or unspecified starting seed.\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--runs") == 0) {
            if (i == argc-1 || sscanf(argv[i+1], "%" PRIu32, &num_runs) != 1) {
                std::fprintf(stderr, "Illegal or unspecified run count.\n");
                return 1;
            }
        }
    }

    std::fprintf(stderr, "Running search, seed=%" PRIu64 ", runs=%" PRIu32 "\n");

    std::uint64_t eqn_seed = start_eqn_seed;
    for (int r = 0; r < num_runs; r++) {
        auto t0 = std::chrono::steady_clock::now();
        run(eqn_seed);
        auto t1 = std::chrono::steady_clock::now();

        for (std::uint32_t i = 0; i < result_count; i++) {
            std::fprintf(stdout, "%" PRIu64 " %" PRIu64 "\n", results[i].lo, results[i].hi);
        }
        eqn_seed++;
    }
}