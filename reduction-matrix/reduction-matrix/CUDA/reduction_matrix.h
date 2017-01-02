#ifndef ReductionMatrixLib_reduction_matrix_H
#define ReductionMatrixLib_reduction_matrix_H

#include <cuda_runtime.h>
#include <cmath>

#include "reduction_definitions.h"
#include "reduction_utilities.h"

#include "../Memory/matrix2D3D.h"

#include "reduction_kernels.h"

namespace ReductionMatrixLib {

	class ReductionMatrix {

		private:
			void static _SumMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static _SumMatrixX(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static _MultiplyMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static _SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static _MultiplyMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

		public:
			void static SumMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				_SumMatrix(inputs, outputs, rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static SumMatrixX(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				_SumMatrixX(inputs, outputs, rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				_MultiplyMatrix(inputs, outputs, rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				_SumMatrix3D(inputs, outputs, rows, columns, planes, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				_MultiplyMatrix3D(inputs, outputs, rows, columns, planes, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}
	};
}
#endif
