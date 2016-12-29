#ifndef ReductionMatrixLib_reduction_kernels_H
#define ReductionMatrixLib_reduction_kernels_H

#include "reduction_definitions.h"

namespace ReductionMatrixLib {

	void KernelSumMatrix(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce);

	void KernelMultiplyMatrix(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int dimensiontoreduce);

	void KernelSumMatrix3D(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce);

	void KernelMultiplyMatrix3D(dim3 blocks, dim3 threads, int blockSize, cudafloat *inputs, cudafloat *outputs, int rows, int columns, int panels, int dimensiontoreduce);
}
#endif
