#ifndef ReductionMatrixLib_reduction_matrix_H
#define ReductionMatrixLib_reduction_matrix_H

#include <cuda_runtime.h>
#include <cmath>

#include "reduction_definitions.h"
#include "reduction_kernels.h"
#include "matrix2D3D.h"

namespace ReductionMatrixLib {

	class ReductionMatrix {

		private:
			void static _TransposeMatrix2D(float *inputs, float *outputs, int rows, int columns);

			void static _SumMatrix2D(float *inputs, float *t_inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int blocksize);

			void static _SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int blocksize);

		public:
			void static TransposeMatrix2D(float *inputs, float *outputs, int rows, int columns) {

				_TransposeMatrix2D(inputs, outputs, rows, columns);
			}

			void static SumMatrix2D(float *inputs, float *t_inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int blocksize) {

				_SumMatrix2D(inputs, t_inputs, outputs, rows, columns, dimensiontoreduce, maxthreadsperblock, blocksize);
			}

			void static SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int blocksize) {

				_SumMatrix3D(inputs, outputs, rows, columns, planes, dimensiontoreduce, maxthreadsperblock, blocksize);
			}
	};

	void ReductionMatrix::_TransposeMatrix2D(float *inputs, float *outputs, int rows, int columns) {

		dim3 threads(TRANSPOSE_MATRIX_TILE_SIZE, TRANSPOSE_MATRIX_TILE_SIZE, 1);

		dim3 blocks(rows / TRANSPOSE_MATRIX_TILE_SIZE, columns / TRANSPOSE_MATRIX_TILE_SIZE, 1);

		KernelTransposeMatrix2D(blocks, threads, inputs, outputs, rows, columns);
	}

	void ReductionMatrix::_SumMatrix2D(float *inputs, float *t_inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int blocksize) {

		dim3 threads(blocksize, 1, 1);

		dim3 blocks((rows * columns) / blocksize, 1, 1);

		if (dimensiontoreduce == 1)
			KernelSumMatrix2D(blocks, threads, inputs, outputs, rows, columns, dimensiontoreduce);
		else {
			_TransposeMatrix2D(inputs, t_inputs, rows, columns);

			cudaDeviceSynchronize();

			KernelSumMatrix2D(blocks, threads, t_inputs, outputs, rows, columns, dimensiontoreduce);
		}
	}

	void ReductionMatrix::_SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int blocksize) {

		dim3 threads(blocksize, blocksize, blocksize);

		dim3 blocks(rows / threads.x, columns / threads.y, planes / threads.z);

		KernelSumMatrix3D(blocks, threads, blocksize, inputs, outputs, rows, columns, planes, dimensiontoreduce);
	}
}
#endif
