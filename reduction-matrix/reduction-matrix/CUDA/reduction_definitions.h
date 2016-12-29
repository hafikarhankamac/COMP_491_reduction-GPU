#ifndef ReductionMatrixLib_reduction_definitions_H
#define ReductionMatrixLib_reduction_definitions_H

#include <float.h>
#include <math.h>

#define USE_SINGLE_PRECISION_VARIABLES

#define USE_SINGLE_PRECISION_FUNCTIONS

#ifndef CUDA_MAX_THREADS_PER_BLOCK

#define CUDA_MAX_THREADS_PER_BLOCK 512

#endif

#define SIZE_SMALL_CUDA_VECTOR (3 * CUDA_MAX_THREADS_PER_BLOCK)

#define OPTIMAL_BLOCK_SIZE_REDUCTION 64

#ifdef USE_SINGLE_PRECISION_VARIABLES

typedef float cudafloat;

#define CUDA_VALUE(X) (X##f)

#define MAX_CUDAFLOAT (FLT_MAX)

#define MIN_POSITIVE_CUDAFLOAT (FLT_MIN)

#define MIN_CUDAFLOAT (-FLT_MAX)

#else

typedef double cudafloat;

#define CUDA_VALUE(X) (X)

#define MAX_CUDAFLOAT (DBL_MAX)

#define MIN_POSITIVE_CUDAFLOAT (DBL_MIN)

#define MIN_CUDAFLOAT (-DBL_MAX)

#endif

namespace ReductionMatrixLib {

#if defined(__CUDA_ARCH__)
	__device__ __forceinline__ bool IsInfOrNaN(cudafloat x) {

		return (isnan(x) || isinf(x));
	}

	__device__ __forceinline__ cudafloat Log(cudafloat x) {
		
		if (x != 0) {
			cudafloat y = log(x);

			if (!IsInfOrNaN(y)) return y;
		}

		return CUDA_VALUE(-7.0);
	}
	
#else
	inline bool IsInfOrNaN(cudafloat x) {
		
#if (defined(_MSC_VER))
		return (!_finite(x));

#else
		return (isnan(x) || isinf(x));
#endif
	}

	inline cudafloat Log(cudafloat x) {
		
		if (x != 0) {
			cudafloat y = log(x);

			if (!IsInfOrNaN(y)) return y;
		}

		return CUDA_VALUE(-7.0);
	}

	inline cudafloat AbsDiff(cudafloat a, cudafloat b) {
		
		cudafloat d = a - b;

		if (d < cudafloat(0.0)) return -d;

		return d;
	}
#endif
}
#endif
