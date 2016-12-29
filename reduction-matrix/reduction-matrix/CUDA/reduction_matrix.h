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
			void static SumMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static MultiplyMatrix(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static SumMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

			void static MultiplyMatrix3D(cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize);

		public:
			void static SumMatrix(DeviceMatrix<cudafloat> &inputs, DeviceMatrix<cudafloat> &outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				SumMatrix(inputs.Pointer(), outputs.Pointer(), rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix(DeviceMatrix<cudafloat> &inputs, DeviceMatrix<cudafloat> &outputs, int rows, int columns, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				MultiplyMatrix(inputs.Pointer(), outputs.Pointer(), rows, columns, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static SumMatrix3D(DeviceMatrix3D<cudafloat> &inputs, DeviceMatrix3D<cudafloat> &outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				SumMatrix3D(inputs.Pointer(), outputs.Pointer(), rows, columns, panels, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}

			void static MultiplyMatrix3D(DeviceMatrix3D<cudafloat> &inputs, DeviceMatrix3D<cudafloat> &outputs, int rows, int columns, int panels, int dimensiontoreduce, int maxthreadsperblock, int optimalblocksize) {

				MultiplyMatrix3D(inputs.Pointer(), outputs.Pointer(), rows, columns, panels, dimensiontoreduce, maxthreadsperblock, optimalblocksize);
			}
	};
}
#endif
