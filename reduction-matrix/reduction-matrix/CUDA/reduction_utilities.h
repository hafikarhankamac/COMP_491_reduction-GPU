#ifndef ReductionMatrixLib_reduction_utilities_H
#define ReductionMatrixLib_reduction_utilities_H

#include "reduction_definitions.h"

namespace ReductionMatrixLib {

	//! Finds the number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
	//! \param threads Number of threads.
	//! \param maxThreadsPerBlock Maximum number of threads.
	//! \return The number of threads (multiple of 2) per block that either is greater that the number of threads needed or identical to the maximum number of threads per block.
	//! \sa CUDA_MAX_THREADS_PER_BLOCK, NumberBlocks
	inline int NumberThreadsPerBlockThatBestFit(int threads, int maxThreadsPerBlock = CUDA_MAX_THREADS_PER_BLOCK) {

		int nt = 1;

		while (nt < threads && nt < maxThreadsPerBlock) nt <<= 1;

		return nt;
	}

	//! Finds the number of blocks needed to execute the number of threads specified, given a block size.
	//! \param threads Number of threads.
	//! \param blockSize Block size.
	//! \return The number of blocks needed to execute the number of threads specified.
	//! \sa NumberThreadsPerBlockThatBestFit, CUDA_MAX_THREADS_PER_BLOCK
	inline int NumberBlocks(int threads, int blockSize) {

		int nb = threads / blockSize;

		if (threads % blockSize != 0) nb++;

		return nb;
	}

	//! Makes sure that the block does not have more than the maximum number of threads supported by CUDA, reducing the number of threads in each dimension if necessary.
	//! \param block block.
	//! \sa MAX_THREADS_PER_BLOCK
	inline void MakeSureBlockDoesNotHaveTooMuchThreads(dim3 & block) {

		unsigned x = NumberThreadsPerBlockThatBestFit(block.x);
		unsigned y = NumberThreadsPerBlockThatBestFit(block.y);
		unsigned z = NumberThreadsPerBlockThatBestFit(block.z);

		while (x * y * z > CUDA_MAX_THREADS_PER_BLOCK) {
			if (z > 1 && z >= y) {
				z >>= 1;
			} else if (y >= x) {
				y >>= 1;
			} else {
				x >>= 1;
			}
		}

		// fix the value of z
		if (z < block.z) block.z = z;

		while (2 * x * y * block.z < CUDA_MAX_THREADS_PER_BLOCK) {
			if (x < block.x) {
				if (y < x && y < block.y) {
					y <<= 1;
				} else {
					x <<= 1;
				}
			} else if (y < block.y) {
				y <<= 1;
			} else {
				break;
			}
		}

		// fix the value of y
		if (y < block.y) block.y = y;

		// fix the value of x
		while (x < block.x && 2 * x * y * z < CUDA_MAX_THREADS_PER_BLOCK) x <<= 1;

		if (x < block.x) block.x = x;
	}
}
#endif
