#ifndef ReductionMatrixLib_reduction_kernels_H
#define ReductionMatrixLib_reduction_kernels_H

#include "reduction_definitions.h"

namespace ReductionMatrixLib {

	void KernelTransposeMatrix2D(dim3 blocks, dim3 threads, float *inputs, float *outputs, int rows, int columns);

	void KernelSumMatrix2D(dim3 blocks, dim3 threads, float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce);

	void KernelSumMatrix3D(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce);

	__device__ volatile float value_of_blocks[MAX_BLOCKS_PER_GRID];

	__global__ void _KernelTransposeMatrix2D(float *inputs, float *outputs, int rows, int columns) {

		__shared__ float value_of_transpose_tile[TRANSPOSE_MATRIX_TILE_SIZE][TRANSPOSE_MATRIX_TILE_SIZE + 1];

		unsigned int blockId_Row;
		unsigned int blockId_Col;

		if (rows == columns) {
			blockId_Col = blockIdx.x;
			blockId_Row = (blockIdx.x + blockIdx.y) % gridDim.x;
		}
		else {
			unsigned int blockId = blockIdx.x + gridDim.x * blockIdx.y;

			blockId_Col = blockId % gridDim.y;
			blockId_Row = ((blockId / gridDim.y) + blockId_Col) % gridDim.x;
		}

		unsigned int inputs_index = (blockId_Row * TRANSPOSE_MATRIX_TILE_SIZE + threadIdx.x) + (blockId_Col * TRANSPOSE_MATRIX_TILE_SIZE + threadIdx.y) * rows;
		unsigned int outputs_index = (blockId_Col * TRANSPOSE_MATRIX_TILE_SIZE + threadIdx.x) + (blockId_Row * TRANSPOSE_MATRIX_TILE_SIZE + threadIdx.y) * columns;

		for (unsigned int i = 0; i < TRANSPOSE_MATRIX_TILE_SIZE; i += TRANSPOSE_MATRIX_TILE_SIZE)
			value_of_transpose_tile[threadIdx.y + i][threadIdx.x] = inputs[inputs_index + i * rows];

		__syncthreads();

		for (unsigned int i = 0; i < TRANSPOSE_MATRIX_TILE_SIZE; i += TRANSPOSE_MATRIX_TILE_SIZE)
			outputs[outputs_index + i * columns] = value_of_transpose_tile[threadIdx.x][threadIdx.y + i];
	}

	void KernelTransposeMatrix2D(dim3 blocks, dim3 threads, float *inputs, float *outputs, int rows, int columns) {

		_KernelTransposeMatrix2D << <blocks, threads >> >(inputs, outputs, rows, columns);
	}

	__global__ void _KernelSumMatrix2DinBlock(float *inputs, int rows, int columns, int size) {

		__shared__ volatile float value_of_shared[MAX_SHARED_MEMORY_SIZE];

		unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;

		value_of_shared[threadIdx.x] = (globalId < size) ? inputs[globalId] : 0;

		__syncthreads();

		//printf("\ngridDim.x = %d, blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, globalId = %d, value_of_shared[threadIdx.x] = %.5f", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, globalId, value_of_shared[threadIdx.x]);

		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
			if (threadIdx.x < i)
				value_of_shared[threadIdx.x] += value_of_shared[threadIdx.x + i];

			__syncthreads();
		}

		if (threadIdx.x == 0) value_of_blocks[blockIdx.x] = value_of_shared[0];

		//if (threadIdx.x == 0)
		//	printf("\ngridDim.x = %d, blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, globalId = %d, value_of_shared[threadIdx.x] = %.5f, value_of_blocks[blockIdx.x] = %.5f", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, globalId, value_of_shared[threadIdx.x], value_of_blocks[blockIdx.x]);
	}

	__global__ void _KernelSumMatrix2DoutBlock(float *outputs, int rows, int columns, int size) {

		__shared__ volatile float value_of_shared[MAX_SHARED_MEMORY_SIZE];

		unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;

		value_of_shared[threadIdx.x] = (globalId < size) ? value_of_blocks[globalId] : 0;

		__syncthreads();

		//printf("\ngridDim.x = %d, blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, globalId = %d, value_of_shared[threadIdx.x] = %.5f", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, globalId, value_of_shared[threadIdx.x]);

		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
			if (threadIdx.x < i)
				value_of_shared[threadIdx.x] += value_of_shared[threadIdx.x + i];

			__syncthreads();
		}

		if (threadIdx.x == 0) outputs[blockIdx.x] = value_of_shared[0];

		//if (threadIdx.x == 0)
		//	printf("\ngridDim.x = %d, blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, globalId = %d, value_of_shared[threadIdx.x] = %.5f, outputs[blockIdx.x] = %.5f", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, globalId, value_of_shared[threadIdx.x], outputs[blockIdx.x]);
	}

	void KernelSumMatrix2D(dim3 blocks, dim3 threads, float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		_KernelSumMatrix2DinBlock << <blocks, threads >> >(inputs, rows, columns, (rows * columns));

		if (dimensiontoreduce == 1) {
			dim3 outthreads(rows / threads.x, 1, 1);
			dim3 outblocks(columns, 1, 1);

			_KernelSumMatrix2DoutBlock << <outblocks, outthreads >> >(outputs, rows, columns, (rows * columns));
		}
		else {
			dim3 outthreads(columns / threads.x, 1, 1);
			dim3 outblocks(rows, 1, 1);

			_KernelSumMatrix2DoutBlock << <outblocks, outthreads >> >(outputs, rows, columns, (rows * columns));
		}
	}

	template <int blockSize> __global__ void _KernelSumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce) {

		unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int Plane = blockIdx.z * blockDim.z + threadIdx.z;

		float subSum = 0.0f;

		if (dimensiontoreduce == 1) {
			for (unsigned int i = 0; i < columns; i++) {
				subSum += inputs[Plane * rows * columns + Row * rows + i];

				__syncthreads();
			}

			outputs[(blockIdx.z * blockSize + threadIdx.z) * columns + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
		if (dimensiontoreduce == 2) {
			for (unsigned int i = 0; i < rows; i++) {
				subSum += inputs[Plane * rows * columns + i * columns + Col];

				__syncthreads();
			}

			outputs[(blockIdx.z * blockSize + threadIdx.z) * rows + blockIdx.x * blockSize + threadIdx.x] = subSum;
		}
		if (dimensiontoreduce == 3) {
			for (unsigned int i = 0; i < planes; i++) {
				subSum += inputs[i * rows * columns + Row * rows + Col];

				__syncthreads();
			}

			outputs[(blockIdx.x * blockSize + threadIdx.x) * planes + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
	}

	void KernelSumMatrix3D(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
			_KernelSumMatrix3D<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 512:
			_KernelSumMatrix3D<512> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 256:
			_KernelSumMatrix3D<256> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 128:
			_KernelSumMatrix3D<128> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 64:
			_KernelSumMatrix3D<64> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 32:
			_KernelSumMatrix3D<32> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 16:
			_KernelSumMatrix3D<16> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 8:
			_KernelSumMatrix3D<8> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 4:
			_KernelSumMatrix3D<4> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 2:
			_KernelSumMatrix3D<2> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 1:
			_KernelSumMatrix3D<1> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		}
	}
}
#endif
