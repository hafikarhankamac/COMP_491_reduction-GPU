#include "reduction_matrix.h"

#include <stdio.h>
#include <math.h>

namespace ReductionMatrixLib {

	void ReductionMatrix::SumMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		//int rowsblockSize = NumberThreadsPerBlockThatBestFit(rows, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int columnsblockSize = NumberThreadsPerBlockThatBestFit(columns, OPTIMAL_BLOCK_SIZE_REDUCTION);

		//int blockSize = ((rowsblockSize >= columnsblockSize) ? rowsblockSize : columnsblockSize);
		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, 1);

		dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), 1);

		KernelSumMatrix(blocks, threads, blockSize, inputs, outputs, rows, columns, dimensiontoreduce);
	}

	void ReductionMatrix::MultiplyMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, 1);

		//dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), 1);
		dim3 blocks(ceil(((rows >= columns) ? rows : columns) / threads.x), ceil(((columns >= rows) ? columns : rows) / threads.y), 1);

		KernelMultiplyMatrix(blocks, threads, blockSize, inputs, outputs, ((rows >= columns) ? rows : columns), ((columns >= rows) ? columns : rows), dimensiontoreduce);
	}

	void ReductionMatrix::SumMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		//int rowsblockSize = NumberThreadsPerBlockThatBestFit(rows, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int columnsblockSize = NumberThreadsPerBlockThatBestFit(columns, OPTIMAL_BLOCK_SIZE_REDUCTION);
		//int panelsblockSize = NumberThreadsPerBlockThatBestFit(panels, OPTIMAL_BLOCK_SIZE_REDUCTION);

		//int blockSize = ((rowsblockSize >= columnsblockSize) ? rowsblockSize : columnsblockSize);
		//blockSize = ((blockSize >= panelsblockSize) ? blockSize : panelsblockSize);
		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, blockSize);

		dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), ceil(panels / threads.z));

		KernelSumMatrix3D(blocks, threads, blockSize, inputs, outputs, rows, columns, panels, dimensiontoreduce);
	}

	void ReductionMatrix::MultiplyMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

		int blockSize = optimalblocksize;

		dim3 threads(blockSize, blockSize, blockSize);

		dim3 blocks(ceil(rows / threads.x), ceil(columns / threads.y), ceil(panels / threads.z));

		KernelMultiplyMatrix3D(blocks, threads, blockSize, inputs, outputs, rows, columns, panels, dimensiontoreduce);
	}
}
