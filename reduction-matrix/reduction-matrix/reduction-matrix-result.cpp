#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#include "./CUDA/reduction_init.h"
#include "./CUDA/reduction_matrix.h"

using namespace ReductionMatrixLib;

#define REDUCTION_METHOD_SUM "sum"
#define REDUCTION_METHOD_MULTIPLY "multiply"

#define SUM_METHOD 1
#define MULTIPLY_METHOD 2

#define TWO_DIMS 2
#define THREE_DIMS 3

#define RANDOM_DATA "random"
#define COLLECT_DATA "collect"

#define RESULT_CONSOLE_OUT "out"

#ifdef WIN32

#define strcasecmp strcmpi

#endif

int main(int argc, char * argv[]) {

	CudaDevice cudadevice;

	if (!cudadevice.SupportsCuda()) {
		printf("\nDevice does not support CUDA...\n");

		return 0;
	}

	if (argc < 8) {
		printf("\nUsage: reduction-matrix\n<reduction method sum or multiply>\n<number of matrix dimension which is 2, 3>\n<number of matrix rows>\n<number of matrix columns>\n<number of matrix planes>\n<matrix dimension number to reduce which is 1, 2, 3>\n<matrix reduction kernel block size 1, 2, 4, 8, 16, 32 or 64>\n<random to initialize with random float data or collect to initialize sequential float data started 1>\n<out to print result to console>\n");

		return 0;
	}

	srand((unsigned int)time(0));

	printf("\n");

	int iReductionMethod = 0;

	if (!strcasecmp(argv[1], REDUCTION_METHOD_SUM))
		iReductionMethod = SUM_METHOD;

	if (!strcasecmp(argv[1], REDUCTION_METHOD_MULTIPLY))
		iReductionMethod = MULTIPLY_METHOD;

	int nDim = atoi(argv[2]);

	int nRows = atoi(argv[3]);

	int nColumns = atoi(argv[4]);

	int nPlanes = 0;

	int nDimReduce = 0;

	int nBlockSize = 0;

	bool bRandomData = false;

	bool bConsoleOut = false;

	if (nDim == TWO_DIMS) {
		nDimReduce = atoi(argv[5]);

		nBlockSize = atoi(argv[6]);

		if (!strcasecmp(argv[7], RANDOM_DATA))
			bRandomData = true;

		if (argc == 9 && !strcasecmp(argv[8], RESULT_CONSOLE_OUT))
			bConsoleOut = true;
	}

	if (nDim == THREE_DIMS) {
		nPlanes = atoi(argv[5]);

		nDimReduce = atoi(argv[6]);

		nBlockSize = atoi(argv[7]);

		if (!strcasecmp(argv[8], RANDOM_DATA))
			bRandomData = true;

		if (argc == 10 && !strcasecmp(argv[9], RESULT_CONSOLE_OUT))
			bConsoleOut = true;
	}

	if (bConsoleOut) // print result to console
		cudadevice.ShowInfo();

	printf("\nExecution parameters\n\n");

	printf("Matrix reduction method                              : %s\n", argv[1]);
	printf("Matrix dimension                                     : %d\n", nDim);

	if (nDim == TWO_DIMS)
		printf("Matrix dimension size (rows, columns)                : (%d, %d)\n", nRows, nColumns);

	if (nDim == THREE_DIMS)
		printf("Matrix dimension size (rows, columns, planes)        : (%d, %d, %d)\n", nRows, nColumns, nPlanes);

	printf("Matrix dimension to reduce                           : %d\n", nDimReduce);
	printf("Matrix reduction kernel block size                   : %d\n", nBlockSize);

	clock_t startTimeGPU, startTimeCPU;

	unsigned elapsedTimeGPU, elapsedTimeCPU;

	int nRowsDim = 0,
		nColumnsDim = 0,
		nPlanesDim = 0;

	int fCollectData = 1;

	cudaError_t cudaError;

	cudaEvent_t cudaEventStart, cudaEventStop;

	if (nDim == TWO_DIMS) { // 2D matrix
		// input matrix
		matrix2D<float> i2DMatrix(nRows, nColumns);

		// initialize input matrix with random or collection values
		for (int i = 0; i < nRows; i++)
			for (int j = 0; j < nColumns; j++) {
				if (!strcasecmp(argv[5], RANDOM_DATA)) {
					i2DMatrix(i, j) = (float)(-1.0) + (float(2.0) * rand()) / RAND_MAX;
				}
				else {
					i2DMatrix(i, j) = (float)fCollectData++;
				}
			}

		// allocate device memory for input matrix
		float *d_i2DMatrix;

		cudaError = cudaMalloc((void **)&d_i2DMatrix, (nRows * nColumns * sizeof(float)));

		if (cudaError != cudaSuccess) {
			printf("\ncudaMalloc input matrix returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy host data to device data for input matrix
		cudaError = cudaMemcpy(d_i2DMatrix, i2DMatrix.Pointer(), (nRows * nColumns * sizeof(float)), cudaMemcpyHostToDevice);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy input matrix to device returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// output matrix
		matrix2D<float> o2DMatrix;

		// temporary matrix for serial CPU code implementation
		matrix2D<float> h2DMatrix;

		if (nDimReduce == 1) {
			nRowsDim = 1;
			nColumnsDim = nColumns;
		}
		if (nDimReduce == 2) {
			nRowsDim = nRows;
			nColumnsDim = 1;
		}

		o2DMatrix.Resize(nRowsDim, nColumnsDim);

		h2DMatrix.Resize(nRowsDim, nColumnsDim);

		// initialize output matrices with zeroes
		for (int i = 0; i < nRowsDim; i++)
			for (int j = 0; j < nColumnsDim; j++) {
				o2DMatrix(i, j) = (float)0;

				h2DMatrix(i, j) = (float)0;
			}

		// allocate device memory for output matrix
		float *d_o2DMatrix;

		cudaError = cudaMalloc((void **)&d_o2DMatrix, (nRowsDim * nColumnsDim * sizeof(float)));

		if (cudaError != cudaSuccess) {
			printf("\ncudaMalloc output matrix returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy host data to device data for output matrix
		cudaError = cudaMemcpy(d_o2DMatrix, o2DMatrix.Pointer(), (nRowsDim * nColumnsDim * sizeof(float)), cudaMemcpyHostToDevice);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy output matrix to device returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// allocate CUDA events that we'll use for timing
		cudaError = cudaEventCreate(&cudaEventStart);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventCreate start event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventCreate(&cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventCreate stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventRecord(cudaEventStart, NULL);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventRecord start event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		startTimeGPU = clock();

		// Execute GPU code
		if (iReductionMethod == SUM_METHOD) // matrix reduction method is sum
			ReductionMatrix::SumMatrix(d_i2DMatrix, d_o2DMatrix, nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);
			//ReductionMatrix::SumMatrixX(d_i2DMatrix, d_o2DMatrix, nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		if (iReductionMethod == MULTIPLY_METHOD) // matrix reduction method is multiply (vector)
			ReductionMatrix::MultiplyMatrix(d_i2DMatrix, d_o2DMatrix, nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		//cudaDeviceSynchronize();

		elapsedTimeGPU = (clock() - startTimeGPU);

		cudaError = cudaEventRecord(cudaEventStop, NULL);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventRecord stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy device data to host data for output matrix
		cudaError = cudaMemcpy(o2DMatrix.Pointer(), d_o2DMatrix, (nRowsDim * nColumnsDim * sizeof(float)), cudaMemcpyDeviceToHost);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy output matrix to host returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventSynchronize(cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventSynchronize stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		float msecTotal = 0.0f;
		cudaError = cudaEventElapsedTime(&msecTotal, cudaEventStart, cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventElapsedTime start and stop events returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// free device memory
		cudaFree(d_o2DMatrix);
		cudaFree(d_i2DMatrix);

		float msecPerMatrixReduction = msecTotal;
		double flopsPerMatrixReduction = (double)nRows * (double)nColumns;
		double gigaFlops = (flopsPerMatrixReduction * 1.0e-9f) / (msecPerMatrixReduction / 1000.0f);

		startTimeCPU = clock();

		// Execute CPU code
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nColumns; j++) {
				if (nDimReduce == 1) {
					h2DMatrix(0, j) += (float)i2DMatrix(j, i);
				}
				if (nDimReduce == 2) {
					h2DMatrix(i, 0) += (float)i2DMatrix(j, i);
				}
			}
		}

		elapsedTimeCPU = (clock() - startTimeCPU);

		if (bConsoleOut) { // print result to console
			printf("\n%dx%d CUDA 2D matrix:\n", nRows, nColumns);

			printf("[:, :] = \n");

			for (int i = 0; i < nRows; i++) {
				for (int j = 0; j < nColumns; j++) {
					printf(" %.5f ", i2DMatrix(j, i));
				}
				printf("\n");
			}

			printf("\n%dx%d reduce dim (%d) matrix:\n", nRowsDim, nColumnsDim, nDimReduce);

			for (int i = 0; i < nRowsDim; i++) {
				for (int j = 0; j < nColumnsDim; j++) {
					printf(" %.5f ", o2DMatrix(i, j));
				}
				printf("\n");
			}
		} // print result to console

		printf("\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops, WorkgroupSize = %u threads/block\n", gigaFlops, msecPerMatrixReduction, flopsPerMatrixReduction, (nRows * nColumns / nBlockSize));

		printf("\nElapsed time GPU = %.3f sec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC);
		printf("\nElapsed time CPU = %.3f sec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC);
	} // 2D matrix

	if (nDim == THREE_DIMS) { // 3D matrix
		// input matrix
		matrix3D<float> i3DMatrix(nRows, nColumns, nPlanes);

		// initialize input matrix with random or collection values
		for (int k = 0; k < nPlanes; k++)
			for (int i = 0; i < nRows; i++)
				for (int j = 0; j < nColumns; j++) {
					if (!strcasecmp(argv[5], RANDOM_DATA)) {
						i3DMatrix(i, j, k) = (float)(-1.0) + (float(2.0) * rand()) / RAND_MAX;
					}
					else {
						i3DMatrix(i, j, k) = (float)fCollectData++;
					}
				}

		// allocate device memory for input matrix
		float *d_i3DMatrix;

		cudaError = cudaMalloc((void **)&d_i3DMatrix, (nRows * nColumns * nPlanes * sizeof(float)));

		if (cudaError != cudaSuccess) {
			printf("\ncudaMalloc input matrix returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy host data to device data for input matrix
		cudaError = cudaMemcpy(d_i3DMatrix, i3DMatrix.Pointer(), (nRows * nColumns * nPlanes * sizeof(float)), cudaMemcpyHostToDevice);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy input matrix to device returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// output matrix
		matrix3D<float> o3DMatrix;

		// temporary matrix for serial CPU code implementation
		matrix3D<float> h3DMatrix;

		if (nDimReduce == 1) {
			nRowsDim = 1;
			nColumnsDim = nColumns;
			nPlanesDim = nPlanes;
		}
		if (nDimReduce == 2) {
			nRowsDim = nRows;
			nColumnsDim = 1;
			nPlanesDim = nPlanes;
		}
		if (nDimReduce == 3) {
			nRowsDim = nRows;
			nColumnsDim = nColumns;
			nPlanesDim = 1;
		}

		o3DMatrix.Resize(nRowsDim, nColumnsDim, nPlanesDim);

		h3DMatrix.Resize(nRowsDim, nColumnsDim, nPlanesDim);

		// initialize output matrices with zeroes
		for (int i = 0; i < nRowsDim; i++)
			for (int j = 0; j < nColumnsDim; j++) {
				for (int k = 0; k < nPlanesDim; k++) {
					o3DMatrix(i, j, k) = (float)0;

					h3DMatrix(i, j, k) = (float)0;
				}
			}

		// allocate device memory for output matrix
		float *d_o3DMatrix;

		cudaError = cudaMalloc((void **)&d_o3DMatrix, (nRowsDim * nColumnsDim * nPlanesDim * sizeof(float)));

		if (cudaError != cudaSuccess) {
			printf("\ncudaMalloc output matrix returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy host data to device data for output matrix
		cudaError = cudaMemcpy(d_o3DMatrix, o3DMatrix.Pointer(), (nRowsDim * nColumnsDim * nPlanesDim * sizeof(float)), cudaMemcpyHostToDevice);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy output matrix to device returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// allocate CUDA events that we'll use for timing
		cudaError = cudaEventCreate(&cudaEventStart);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventCreate start event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventCreate(&cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventCreate stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventRecord(cudaEventStart, NULL);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventRecord start event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		startTimeGPU = clock();

		// Execute GPU code
		if (iReductionMethod == SUM_METHOD) // matrix reduction method is sum
			ReductionMatrix::SumMatrix3D(d_i3DMatrix, d_o3DMatrix, nRows, nColumns, nPlanes, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		if (iReductionMethod == MULTIPLY_METHOD) // matrix reduction method is multiply (vector)
			ReductionMatrix::MultiplyMatrix3D(d_i3DMatrix, d_o3DMatrix, nRows, nColumns, nPlanes, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		//cudaDeviceSynchronize();

		elapsedTimeGPU = (clock() - startTimeGPU);

		cudaError = cudaEventRecord(cudaEventStop, NULL);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventRecord stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// copy device data to host data for output matrix
		cudaError = cudaMemcpy(o3DMatrix.Pointer(), d_o3DMatrix, (nRowsDim * nColumnsDim * nPlanesDim * sizeof(float)), cudaMemcpyDeviceToHost);

		if (cudaError != cudaSuccess) {
			printf("\ncudaMemcpy output matrix to host returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		cudaError = cudaEventSynchronize(cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventSynchronize stop event returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		float msecTotal = 0.0f;
		cudaError = cudaEventElapsedTime(&msecTotal, cudaEventStart, cudaEventStop);

		if (cudaError != cudaSuccess) {
			printf("\ncudaEventElapsedTime start and stop events returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);

			exit(EXIT_FAILURE);
		}

		// free device memory
		cudaFree(d_o3DMatrix);
		cudaFree(d_i3DMatrix);

		float msecPerMatrixReduction = msecTotal;
		double flopsPerMatrixReduction = (double)nRows * (double)nColumns * (double)nPlanes;
		double gigaFlops = (flopsPerMatrixReduction * 1.0e-9f) / (msecPerMatrixReduction / 1000.0f);

		startTimeCPU = clock();

		// Execute CPU code
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nColumns; j++) {
				for (int k = 0; k < nPlanes; k++) {
					if (nDimReduce == 1) {
						h3DMatrix(0, j, k) += (float)i3DMatrix(j, i, k);
					}
					if (nDimReduce == 2) {
						h3DMatrix(i, 0, k) += (float)i3DMatrix(j, i, k);
					}
					if (nDimReduce == 3) {
						h3DMatrix(i, j, 0) += (float)i3DMatrix(j, i, k);
					}
				}
			}
		}

		elapsedTimeCPU = (clock() - startTimeCPU);

		if (bConsoleOut) { // print result to console
			printf("\n%dx%dx%d CUDA 3D matrix:\n", nRows, nColumns, nPlanes);

			for (int k = 0; k < nPlanes; k++) {
				printf("[:, :, %d] = \n", k + 1);
				for (int i = 0; i < nRows; i++) {
					for (int j = 0; j < nColumns; j++) {
						printf(" %.5f ", i3DMatrix(j, i, k));
					}
					printf("\n");
				}
				printf("\n");
			}

			printf("\n%dx%dx%d reduce dim (%d) matrix:\n", nRowsDim, nColumnsDim, nPlanesDim, nDimReduce);

			for (int k = 0; k < nPlanesDim; k++) {
				printf("[:, :, %d] = \n", k + 1);
				for (int i = 0; i < nRowsDim; i++) {
					for (int j = 0; j < nColumnsDim; j++) {
						printf(" %.5f ", o3DMatrix(i, j, k));
					}
					printf("\n");
				}
				printf("\n");
			}
		} // print result to console

		printf("\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops, WorkgroupSize = %u threads/block\n", gigaFlops, msecPerMatrixReduction, flopsPerMatrixReduction, (nRows * nColumns * nPlanes / nBlockSize));

		printf("\nElapsed time GPU = %.3f sec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC);
		printf("\nElapsed time CPU = %.3f sec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC);
	} // 3D matrix

	return 0;
}
