#include "reduction_kernels.h"
#include "reduction_warp.h"

#include <stdio.h>

namespace ReductionMatrixLib {

	template <int blockSize> __global__ void _KernelSumMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce) {

		cudafloat SumSub = 0.0f;

		int Row = blockIdx.y * blockDim.y + threadIdx.y;

		int Col = blockIdx.x * blockDim.x + threadIdx.x;


		if (dimensiontoreduce == 1) {
			for (int i = 0; i < rows; i++) {
				SumSub += inputs[Row * rows + i];

				__syncthreads();
			}

			if (threadIdx.x == 0) {
				outputs[blockIdx.y * blockSize + threadIdx.y] = SumSub;
			}

		}
		if (dimensiontoreduce == 2) {
			for (int i = 0; i < columns; i++) {
				SumSub += inputs[i * columns + Col];

				__syncthreads();
			}
			
			if (threadIdx.y == 0) {
				outputs[blockIdx.x * blockSize + threadIdx.x] = SumSub;
			}
		}
	}


	void KernelSumMatrix(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce) {

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

	template <int blockSize> __global__ void _KernelMultiplyMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce) {

		cudafloat SumSub = 0.0f;

		cudafloat *subinputs;

		int Row = threadIdx.y;
		int Col = threadIdx.x;

		if (dimensiontoreduce == 1) {
			for (int i = 0; i < (rows / blockSize); ++i) {
				__shared__ cudafloat sharedinputs[blockSize * blockSize];

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
				__shared__ cudafloat sharedinputs[blockSize * blockSize];

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

	void KernelMultiplyMatrix(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce) {

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

	template <int blockSize> __global__ void _KernelSumMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce) {

		cudafloat subSum = 0.0f;

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
			for (int i = 0; i < panels; i++) {
				subSum += inputs[i * rows * columns + Row * rows + Col];

				__syncthreads();
			}

			outputs[(blockIdx.x * blockSize + threadIdx.x) * panels + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
	}

	void KernelSumMatrix3D(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
			_KernelSumMatrix3D<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 512:
			_KernelSumMatrix3D<512> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 256:
			_KernelSumMatrix3D<256> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 128:
			_KernelSumMatrix3D<128> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 64:
			_KernelSumMatrix3D<64> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 32:
			_KernelSumMatrix3D<32> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 16:
			_KernelSumMatrix3D<16> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 8:
			_KernelSumMatrix3D<8> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 4:
			_KernelSumMatrix3D<4> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 2:
			_KernelSumMatrix3D<2> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 1:
			_KernelSumMatrix3D<1> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		}
	}

	template <int blockSize> __global__ void _KernelMultiplyMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce) {

		cudafloat subSum = 0.0f;

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
			for (int i = 0; i < panels; i++) {
				subSum += inputs[i * rows * columns + Row * rows + Col];

				__syncthreads();
			}

			outputs[(blockIdx.x * blockSize + threadIdx.x) * panels + blockIdx.y * blockSize + threadIdx.y] = subSum;
		}
	}

	void KernelMultiplyMatrix3D(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce) {

		switch (blockSize) {
		case 1024:
		//	_KernelMultiplyMatrix3D<1024> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
		//	break;
		case 512:
		//	_KernelMultiplyMatrix3D<512> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
		//	break;
		case 256:
		//	_KernelMultiplyMatrix3D<256> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
		//	break;
		case 128:
		//	_KernelMultiplyMatrix3D<128> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
		//	break;
		case 64:
			_KernelMultiplyMatrix3D<64> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 32:
			_KernelMultiplyMatrix3D<32> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 16:
			_KernelMultiplyMatrix3D<16> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 8:
			_KernelMultiplyMatrix3D<8> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 4:
			_KernelMultiplyMatrix3D<4> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 2:
			_KernelMultiplyMatrix3D<2> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		case 1:
			_KernelMultiplyMatrix3D<1> << <blocks, threads >> >(inputs, outputs, rows, columns, panels, dimensiontoreduce);
			break;
		}
	}
}
