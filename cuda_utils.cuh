#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__host__ static void gpuAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d — %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

#endif