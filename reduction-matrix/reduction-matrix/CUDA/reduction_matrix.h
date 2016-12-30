#ifndef ReductionMatrixLib_reduction_matrix_H
#define ReductionMatrixLib_reduction_matrix_H

#include <cuda_runtime.h>
#include <cmath>

#include "reduction_definitions.h"
#include "reduction_utilities.h"

#include "../Memory/CUDAMatrix.h"
#include "../Memory/CUDAMatrix3D.h"
#include "../Memory/DeviceAccessibleVariable.h"

#include "reduction_kernels.h"

namespace ReductionMatrixLib {

	class ReductionMatrix {

		private:
			void static SumMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static SumMatrixX(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static MultiplyMatrix(float *inputs, float *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static SumMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static MultiplyMatrix3D(float *inputs, float *outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

		public:
			void static SumMatrix(DeviceMatrix<float> &inputs, DeviceMatrix<float> &outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				SumMatrix(inputs.Pointer(), outputs.Pointer(), rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static SumMatrixX(DeviceMatrix<float> &inputs, DeviceMatrix<float> &outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				SumMatrixX(inputs.Pointer(), outputs.Pointer(), rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix(DeviceMatrix<float> &inputs, DeviceMatrix<float> &outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				MultiplyMatrix(inputs.Pointer(), outputs.Pointer(), rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static SumMatrix3D(DeviceMatrix3D<float> &inputs, DeviceMatrix3D<float> &outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				SumMatrix3D(inputs.Pointer(), outputs.Pointer(), rows, columns, planes, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix3D(DeviceMatrix3D<float> &inputs, DeviceMatrix3D<float> &outputs, int rows, int columns, int planes, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				MultiplyMatrix3D(inputs.Pointer(), outputs.Pointer(), rows, columns, planes, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}
	};
}
#endif
