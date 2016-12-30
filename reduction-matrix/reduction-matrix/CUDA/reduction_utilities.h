#ifndef ReductionMatrixLib_reduction_utilities_H
#define ReductionMatrixLib_reduction_utilities_H

#include "reduction_definitions.h"

namespace ReductionMatrixLib {

	/*
	void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int maxGridSizeX, int maxThreadsPerBlock, int &blocks, int &threads) {

		if (whichKernel < 3) {
			threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
			blocks = (n + threads - 1) / threads;
		}
		else {
			threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
			blocks = (n + (threads * 2 - 1)) / (threads * 2);
		}

		if ((float)(threads * blocks) > (float)(maxGridSizeX * maxThreadsPerBlock)) {
			//printf("n is too large, please choose a smaller number!\n");
		}

		if (blocks > maxGridSizeX) {
			//printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n", blocks, maxGridSizeX, threads * 2, threads);

			blocks /= 2;
			threads *= 2;
		}

		if (whichKernel == 6) {
			blocks = MIN(maxBlocks, blocks);
		}
	}
	*/

	inline int NumberThreadsPerBlockThatBestFit(int threads, int maxThreadsPerBlock = CUDA_MAX_THREADS_PER_BLOCK) {

		int nt = 1;

		while (nt < threads && nt < maxThreadsPerBlock) nt <<= 1;

		return nt;
	}

	inline int NumberBlocks(int threads, int blockSize) {

		int nb = threads / blockSize;

		if (threads % blockSize != 0) nb++;

		return nb;
	}

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
