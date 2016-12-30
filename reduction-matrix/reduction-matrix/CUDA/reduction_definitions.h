#ifndef ReductionMatrixLib_reduction_definitions_H
#define ReductionMatrixLib_reduction_definitions_H

#include <float.h>
#include <math.h>

#ifndef MIN

#define MIN(x,y) ((x < y) ? x : y)

#endif

//#define CUDA_MAX_THREADS_PER_BLOCK 512
#define CUDA_MAX_THREADS_PER_BLOCK 1024

#define SIZE_SMALL_CUDA_VECTOR (3 * CUDA_MAX_THREADS_PER_BLOCK)

#define OPTIMAL_BLOCK_SIZE_REDUCTION 128

#define OPTIMAL_SHARED_MEMORY_SIZE 8192

#endif
