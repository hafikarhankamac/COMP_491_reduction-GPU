#include "reduction_kernels.h"
#include "reduction_warp.h"

#include <stdio.h>
#include <cuda_runtime.h>

namespace ReductionMatrixLib {

	template <int blockSize> __global__ void _KernelSumMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		float SumSub = 0.0f;

		int Row = blockIdx.y * blockDim.y + threadIdx.y;

		int Col = blockIdx.x * blockDim.x + threadIdx.x;

		if (dimensiontoreduce == 1) {
			for (int i = 0; i < rows; i++) {
				SumSub += inputs[Row * rows + i];

				__syncthreads();
			}

			outputs[blockIdx.y * blockSize + threadIdx.y] = SumSub;
		}
		if (dimensiontoreduce == 2) {
			for (int i = 0; i < columns; i++) {
				SumSub += inputs[i * columns + Col];

				__syncthreads();
			}

			outputs[blockIdx.x * blockSize + threadIdx.x] = SumSub;
		}
	}

	void KernelSumMatrix(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
			_KernelSumMatrix<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 512:
			_KernelSumMatrix<512> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 256:
			_KernelSumMatrix<256> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 128:
			_KernelSumMatrix<128> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 64:
			_KernelSumMatrix<64> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 32:
			_KernelSumMatrix<32> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 16:
			_KernelSumMatrix<16> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 8:
			_KernelSumMatrix<8> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 4:
			_KernelSumMatrix<4> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 2:
			_KernelSumMatrix<2> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 1:
			_KernelSumMatrix<1> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		}
	}

	template <int blockSize> __global__ void _KernelSumMatrixX(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		//__shared__ volatile float sharedinputs[CUDA_MAX_THREADS_PER_BLOCK];
		__shared__ volatile float sharedinputs[OPTIMAL_SHARED_MEMORY_SIZE];

		int tId = threadIdx.y * blockDim.y + threadIdx.x;

		sharedinputs[tId] = (globalId < (rows * columns)) ? inputs[globalId] : 0;

		__syncthreads();

		//printf("\ngridDim.x = %d, gridDim.y = %d, blockDim.x = %d, blockDim.y = %d, blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, threadIdx.y = %d, GlobalId = %d, tId = %d, sharedinputs[tId] = %.5f", gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, globalId, tId, sharedinputs[tId]);

		if (dimensiontoreduce == 1) {
			if (globalId % columns == 0) {
				for (int j = 1;  j < columns; j++) {
					sharedinputs[tId] += sharedinputs[tId + j];
				}

				__syncthreads();
			}

			//printf("\ngridDim.x = %d, gridDim.y = %d, blockDim.x = %d, blockDim.y = %d, blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, threadIdx.y = %d, GlobalId = %d, tId = %d, sharedinputs[tId] = %.5f", gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, globalId, tId, sharedinputs[tId]);

			if (tId % columns == 0) outputs[globalId / columns] = sharedinputs[tId];
		}

		if (dimensiontoreduce == 2) {
			for (int i = 0; i < rows; i++) {
				if (globalId % i == 0) {
					sharedinputs[tId] += sharedinputs[tId + i + 1];
				}

				__syncthreads();
			}

			//printf("\ngridDim.x = %d, gridDim.y = %d, blockDim.x = %d, blockDim.y = %d, blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, threadIdx.y = %d, GlobalId = %d, tId = %d, sharedinputs[tId] = %.5f", gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, globalId, tId, sharedinputs[tId]);

			if (tId % rows == 0) outputs[globalId / rows] = sharedinputs[tId];
		}
	}

	void KernelSumMatrixX(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
			_KernelSumMatrixX<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 512:
			_KernelSumMatrixX<512> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 256:
			_KernelSumMatrixX<256> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 128:
			_KernelSumMatrixX<128> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 64:
			_KernelSumMatrixX<64> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 32:
			_KernelSumMatrixX<32> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 16:
			_KernelSumMatrixX<16> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 8:
			_KernelSumMatrixX<8> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 4:
			_KernelSumMatrixX<4> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 2:
			_KernelSumMatrixX<2> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 1:
			_KernelSumMatrixX<1> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		}
	}

	template <int blockSize> __global__ void _KernelMultiplyMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		float SumSub = 0.0f;

		float *subinputs;

		int Row = threadIdx.y;
		int Col = threadIdx.x;

		if (dimensiontoreduce == 1) {
			for (int i = 0; i < (rows / blockSize); ++i) {
				__shared__ float sharedinputs[blockSize * blockSize];

				subinputs = inputs + blockSize * (blockIdx.y + i * rows);

				if (Row < blockSize && Col < blockSize) {
					sharedinputs[Col + Row * blockSize] = subinputs[Col + Row * rows];

					__syncthreads();
				}
				for (int j = 0; j < blockSize; ++j) {
					if (Row < blockSize)
						SumSub += sharedinputs[j + Row * blockSize];
				}

				__syncthreads();
			}

			if (Row < blockSize)
				outputs[blockIdx.y * blockSize + Row] = SumSub;
		}
		if (dimensiontoreduce == 2) {
			for (int i = 0; i < (columns / blockSize); ++i) {
				__shared__ float sharedinputs[blockSize * blockSize];

				subinputs = inputs + blockSize * (blockIdx.x + i * columns);

				if (Row < blockSize && Col < blockSize) {
					sharedinputs[Row + Col * blockSize] = subinputs[Row + Col * columns];

					__syncthreads();
				}
				for (int j = 0; j < blockSize; ++j) {
					if (Col < blockSize)
						SumSub += sharedinputs[Col + j * blockSize];
				}

				__syncthreads();
			}

			if (Col < blockSize)
				outputs[blockIdx.x * blockSize + Col] = SumSub;
		}
	}

	void KernelMultiplyMatrix(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
		//	_KernelMultiplyMatrix<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
		//	break;
		case 512:
		//	_KernelMultiplyMatrix<512> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
		//	break;
		case 256:
		//	_KernelMultiplyMatrix<256> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
		//	break;
		case 128:
		//	_KernelMultiplyMatrix<128> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
		//	break;
		case 64:
			_KernelMultiplyMatrix<64> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 32:
			_KernelMultiplyMatrix<32> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 16:
			_KernelMultiplyMatrix<16> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 8:
			_KernelMultiplyMatrix<8> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 4:
			_KernelMultiplyMatrix<4> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 2:
			_KernelMultiplyMatrix<2> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		case 1:
			_KernelMultiplyMatrix<1> << <blocks, threads >> >(inputs, outputs, rows, columns, dimensiontoreduce);
			break;
		}
	}

	template <int blockSize> __global__ void _KernelSumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce) {

		float subSum = 0.0f;

		int Row = blockIdx.y * blockDim.y + threadIdx.y;

		int Col = blockIdx.x * blockDim.x + threadIdx.x;

		int Plane = blockIdx.z * blockDim.z + threadIdx.z;

		//printf("\nblockDim.x = %d, blockDim.y = %d, blockDim.z = %d, blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, threadIdx.x = %d, threadIdx.y = %d threadIdx.z = %d\nPlane = %d Row = %d, Col = %d, Index = %d", blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, Plane, Row, Col, (Plane * rows * columns + Row * rows + Col));

		if (dimensiontoreduce == 1) {
			for (int i = 0; i < rows; i++) {
				subSum += inputs[Plane * rows * columns + i * columns + Col];

				__syncthreads();
			}
	
			outputs[(blockIdx.z * blockSize + threadIdx.z) * rows + blockIdx.x * blockSize + threadIdx.x] = subSum;
		}
		if (dimensiontoreduce == 2) {
			for (int i = 0; i < columns; i++) {
				subSum += inputs[Plane * rows * columns + Row * rows + i];

				__syncthreads();
			}

			outputs[(blockIdx.z * blockSize + threadIdx.z) * columns + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
		if (dimensiontoreduce == 3) {
			for (int i = 0; i < planes; i++) {
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

	template <int blockSize> __global__ void _KernelMultiplyMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce) {

		float subSum = 0.0f;

		int Row = blockIdx.y * blockDim.y + threadIdx.y;

		int Col = blockIdx.x * blockDim.x + threadIdx.x;

		int Plane = blockIdx.z * blockDim.z + threadIdx.z;

		if (dimensiontoreduce == 1) {
			for (int i = 0; i < rows; i++) {
				subSum += inputs[Plane * rows * columns + i * columns + Col];

				__syncthreads();
			}

			outputs[(blockIdx.z * blockSize + threadIdx.z) * rows + blockIdx.x * blockSize + threadIdx.x] = subSum;
		}
		if (dimensiontoreduce == 2) {
			for (int i = 0; i < columns; i++) {
				subSum += inputs[Plane * rows * columns + Row * rows + i];

				__syncthreads();
			}

			outputs[(blockIdx.z * blockSize + threadIdx.z) * columns + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
		if (dimensiontoreduce == 3) {
			for (int i = 0; i < planes; i++) {
				subSum += inputs[i * rows * columns + Row * rows + Col];

				__syncthreads();
			}

			outputs[(blockIdx.x * blockSize + threadIdx.x) * planes + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
	}

	void KernelMultiplyMatrix3D(dim3 blocks, dim3 threads, int blockSize, float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
		//	_KernelMultiplyMatrix3D<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
		//	break;
		case 512:
		//	_KernelMultiplyMatrix3D<512> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
		//	break;
		case 256:
		//	_KernelMultiplyMatrix3D<256> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
		//	break;
		case 128:
		//	_KernelMultiplyMatrix3D<128> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
		//	break;
		case 64:
			_KernelMultiplyMatrix3D<64> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 32:
			_KernelMultiplyMatrix3D<32> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 16:
			_KernelMultiplyMatrix3D<16> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 8:
			_KernelMultiplyMatrix3D<8> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 4:
			_KernelMultiplyMatrix3D<4> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 2:
			_KernelMultiplyMatrix3D<2> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		case 1:
			_KernelMultiplyMatrix3D<1> << <blocks, threads >> >(inputs, outputs, rows, columns, planes, dimensiontoreduce);
			break;
		}
	}
}
