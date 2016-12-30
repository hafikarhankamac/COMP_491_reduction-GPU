#include "reduction_matrix.h"

#include <stdio.h>
#include <math.h>

namespace ReductionMatrixLib {

	void ReductionMatrix::SumMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		//int rowsblockSize = NumberThreadsPerBlockThatBestFit(rows, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int columnsblockSize = NumberThreadsPerBlockThatBestFit(columns, OPTIMAL_BLOCK_SIZE_REDUCTION);

		//int blockSize = ((rowsblockSize >= columnsblockSize) ? rowsblockSize : columnsblockSize);
		int blockSize = optimalblocksize;

		//dim3 threads(blockSize * blockSize, 1, 1);
		dim3 threads(blockSize, blockSize, 1);

		//dim3 blocks((rows / threads.x) * (columns / threads.y), 1, 1);
		//dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), 1);
		dim3 blocks(rows / threads.x, columns / threads.y, 1);

		KernelSumMatrix(blocks, threads, blockSize, inputs, outputs, rows, columns, dimensiontoreduce);
	}

	void ReductionMatrix::SumMatrixX(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		//int rowsblockSize = NumberThreadsPerBlockThatBestFit(rows, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int columnsblockSize = NumberThreadsPerBlockThatBestFit(columns, OPTIMAL_BLOCK_SIZE_REDUCTION);

		//int blockSize = ((rowsblockSize >= columnsblockSize) ? rowsblockSize : columnsblockSize);
		int blockSize = optimalblocksize;

		//dim3 threads(blockSize * blockSize, 1, 1);
		dim3 threads(blockSize, blockSize, 1);

		//dim3 threads(blockSize, 1, 1);
		//dim3 threads(128, 1);

		//dim3 blocks((rows / threads.x) * (columns / threads.y), 1, 1);
		//dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), 1);
		dim3 blocks(rows / threads.x, columns / threads.y, 1);

		//dim3 blocks((rows / 128 + 1), columns);

		//printf("\nblocks.x = %d, blocks.y= %d, threads.x = %d, threads.y = %d\n", blocks.x, blocks.y, threads.x, threads.y);

		KernelSumMatrixX(blocks, threads, blockSize, inputs, outputs, rows, columns, dimensiontoreduce);
	}

	void ReductionMatrix::MultiplyMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, 1);

		//dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), 1);
		dim3 blocks(ceil(((rows >= columns) ? rows : columns) / threads.x), ceil(((columns >= rows) ? columns : rows) / threads.y), 1);

		KernelMultiplyMatrix(blocks, threads, blockSize, inputs, outputs, ((rows >= columns) ? rows : columns), ((columns >= rows) ? columns : rows), dimensiontoreduce);
	}

	void ReductionMatrix::SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		//int rowsblockSize = NumberThreadsPerBlockThatBestFit(rows, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int columnsblockSize = NumberThreadsPerBlockThatBestFit(columns, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int panelsblockSize = NumberThreadsPerBlockThatBestFit(planes, OPTIMAL_BLOCK_SIZE_REDUCTION);

		//int blockSize = ((rowsblockSize >= columnsblockSize) ? rowsblockSize : columnsblockSize);
		//blockSize = ((blockSize >= panelsblockSize) ? blockSize : panelsblockSize);
		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, blockSize);

		dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), ceil(planes / threads.z));

		KernelSumMatrix3D(blocks, threads, blockSize, inputs, outputs, rows, columns, planes, dimensiontoreduce);
	}

	void ReductionMatrix::MultiplyMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, blockSize);

		dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), ceil(planes / threads.z));

		KernelMultiplyMatrix3D(blocks, threads, blockSize, inputs, outputs, rows, columns, planes, dimensiontoreduce);
	}
}
