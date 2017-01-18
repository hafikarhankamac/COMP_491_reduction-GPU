#ifndef ReductionMatrixLib_reduction_definitions_H
#define ReductionMatrixLib_reduction_definitions_H

#include <float.h>
#include <math.h>

#ifndef MIN

#define MIN(x, y) ((x < y) ? x : y)

#endif

#ifndef MAX

#define MAX(x, y) (x > y ? x : y)

#endif

#define MAX_BLOCKS_PER_GRID 32768

//#define MAX_THREADS_PER_BLOCK 512

#define MAX_THREADS_PER_BLOCK 1024

#define MAX_THREADS_PER_BLOCK_IN_3D 8

#define MAX_SHARED_MEMORY_SIZE 1024

#define TRANSPOSE_MATRIX_TILE_SIZE 16

#endif
