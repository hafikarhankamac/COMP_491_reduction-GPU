#ifndef ReductionMatrixLib_reduction_warp_h
#define ReductionMatrixLib_reduction_warp_h

#include "reduction_definitions.h"

namespace ReductionMatrixLib {

	template <int blockSize> __device__ __forceinline__ void SumBeforeWarp(float * s) {

		if (blockSize >= 1024) {
			if (threadIdx.x < 512) s[threadIdx.x] += s[threadIdx.x + 512];

			__syncthreads();
		}

		if (blockSize >= 512) {
			if (threadIdx.x < 256) s[threadIdx.x] += s[threadIdx.x + 256];

			__syncthreads();
		}

		if (blockSize >= 256) {
			if (threadIdx.x < 128) s[threadIdx.x] += s[threadIdx.x + 128];

			__syncthreads();
		}

		if (blockSize >= 128) {
			if (threadIdx.x < 64) s[threadIdx.x] += s[threadIdx.x + 64];

			__syncthreads();
		}
	}

	template <int blockSize> __device__ __forceinline__ void SumWarp(volatile float * s) {

		if (blockSize >= 64) s[threadIdx.x] += s[threadIdx.x + 32];
		if (blockSize >= 32) s[threadIdx.x] += s[threadIdx.x + 16];
		if (blockSize >= 16) s[threadIdx.x] += s[threadIdx.x + 8];
		if (blockSize >= 8) s[threadIdx.x] += s[threadIdx.x + 4];
		if (blockSize >= 4) s[threadIdx.x] += s[threadIdx.x + 2];
		if (blockSize >= 2) s[threadIdx.x] += s[threadIdx.x + 1];
	}
}
#endif
