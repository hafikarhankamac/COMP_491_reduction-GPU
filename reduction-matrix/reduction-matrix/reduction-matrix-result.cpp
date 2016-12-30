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

	cudadevice.ShowInfo();

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

	printf("\nExecution parameters\n\n");
	printf("Matrix reduction method                              : %s\n", argv[1]);
	printf("Matrix dimension                                     : %d\n", nDim);

	if (nDim == TWO_DIMS)
		printf("Matrix dimension size (rows, columns)                : (%d, %d)\n", nRows, nColumns);

	if (nDim == THREE_DIMS)
		printf("Matrix dimension size (rows, columns, planes)        : (%d, %d, %d)\n", nRows, nColumns, nPlanes);

	printf("Matrix dimension to reduce                           : %d\n", nDimReduce);
	printf("Matrix reduction kernel block size                   : %d\n", nBlockSize);

	clock_t startTimeGPU;

	unsigned elapsedTimeGPU;

	clock_t startTimeCPU;

	unsigned elapsedTimeCPU;

	cudaError_t error;

	int nRowsDim = 0,
		nColumnsDim = 0,
		nPlanesDim = 0;

	int fCollectData = 1;

	cudaEvent_t eventStart;

	cudaEvent_t eventStop;

	if (nDim == TWO_DIMS) { // 2D matrix
		CudaMatrix<float> i2DMatrix;

		i2DMatrix.ResizeWithoutPreservingData(nRows, nColumns);

		// initialize CUDA matrix for device with random or collection values
		for (int i = 0; i < nRows; i++)
			for (int j = 0; j < nColumns; j++) {
				if (!strcasecmp(argv[5], RANDOM_DATA)) {
					i2DMatrix(i, j) = (float)(-1.0) + (float(2.0) * rand()) / RAND_MAX;
				}
				else {
					i2DMatrix(i, j) = (float)fCollectData++;
				}
			}

		// update device data with host data
		i2DMatrix.UpdateDevice();

		CudaMatrix<float> o2DMatrix;

		HostMatrix<float> h2DMatrix;

		if (nDimReduce == 1) {
			nRowsDim = 1;
			nColumnsDim = nColumns;
		}
		if (nDimReduce == 2) {
			nRowsDim = nRows;
			nColumnsDim = 1;
		}

		o2DMatrix.ResizeWithoutPreservingData(nRowsDim, nColumnsDim);

		h2DMatrix.ResizeWithoutPreservingData(nRowsDim, nColumnsDim);

		// initialize CUDA and Host output matrices with zeroes
		for (int i = 0; i < nRowsDim; i++)
			for (int j = 0; j < nColumnsDim; j++) {
				o2DMatrix(i, j) = (float)0;

				h2DMatrix(i, j) = (float)0;
			}

		// update device data with host data
		o2DMatrix.UpdateDevice();

		// allocate CUDA events that we'll use for timing
		error = cudaEventCreate(&eventStart);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventCreate(&eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventRecord(eventStart, NULL);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		startTimeGPU = clock();

		// Execute GPU code
		if (iReductionMethod == SUM_METHOD) // matrix reduction method is sum
			//ReductionMatrix::SumMatrix(i2DMatrix.GetDeviceMatrix(), o2DMatrix.GetDeviceMatrix(), nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);
			ReductionMatrix::SumMatrixX(i2DMatrix.GetDeviceMatrix(), o2DMatrix.GetDeviceMatrix(), nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		if (iReductionMethod == MULTIPLY_METHOD) // matrix reduction method is multiply (vector)
			ReductionMatrix::MultiplyMatrix(i2DMatrix.GetDeviceMatrix(), o2DMatrix.GetDeviceMatrix(), nRows, nColumns, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		//cudaDeviceSynchronize();

		// update host data with device data
		o2DMatrix.UpdateHost();

		elapsedTimeGPU = (clock() - startTimeGPU);

		error = cudaEventRecord(eventStop, NULL);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventSynchronize(eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		float msecTotal = 0.0f;
		error = cudaEventElapsedTime(&msecTotal, eventStart, eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		float msecPerMatrixMul = msecTotal;
		//double flopsPerMatrixMul = 2.0 * (double)nRows * (double)nColumns;
		double flopsPerMatrixMul = (double)nRows * (double)nColumns;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

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

		printf("\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops, WorkgroupSize = %u threads/block\n",
			gigaFlops,
			msecPerMatrixMul,
			flopsPerMatrixMul,
			(nRows * nColumns / nBlockSize));

		//printf("\nElapsed time GPU = %.3f msec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC / 1000);
		//printf("\nElapsed time CPU = %.3f msec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC / 1000);
		printf("\nElapsed time GPU = %.3f sec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC);
		printf("\nElapsed time CPU = %.3f sec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC);
	} // 2D matrix

	if (nDim == THREE_DIMS) { // 3D matrix
		CudaMatrix3D<float> i3DMatrix;

		i3DMatrix.ResizeWithoutPreservingData(nRows, nColumns, nPlanes);

		// initialize CUDA matrix for device with random or collection values
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

		// update device data with host data
		i3DMatrix.UpdateDevice();

		CudaMatrix3D<float> o3DMatrix;

		HostMatrix3D<float> h3DMatrix;

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

		o3DMatrix.ResizeWithoutPreservingData(nRowsDim, nColumnsDim, nPlanesDim);

		h3DMatrix.ResizeWithoutPreservingData(nRowsDim, nColumnsDim, nPlanesDim);

		// initialize CUDA and Host output matrices with zeroes
		for (int i = 0; i < nRowsDim; i++)
			for (int j = 0; j < nColumnsDim; j++) {
				for (int k = 0; k < nPlanesDim; k++) {
					o3DMatrix(i, j, k) = (float)0;

					h3DMatrix(i, j, k) = (float)0;
				}
			}

		// update device data with host data
		o3DMatrix.UpdateDevice();

		// allocate CUDA events that we'll use for timing
		error = cudaEventCreate(&eventStart);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventCreate(&eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventRecord(eventStart, NULL);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		startTimeGPU = clock();

		// Execute GPU code
		if (iReductionMethod == SUM_METHOD) // matrix reduction method is sum
			ReductionMatrix::SumMatrix3D(i3DMatrix.Get3DDeviceMatrix(), o3DMatrix.Get3DDeviceMatrix(), nRows, nColumns, nPlanes, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		if (iReductionMethod == MULTIPLY_METHOD) // matrix reduction method is multiply (vector)
			ReductionMatrix::MultiplyMatrix3D(i3DMatrix.Get3DDeviceMatrix(), o3DMatrix.Get3DDeviceMatrix(), nRows, nColumns, nPlanes, nDimReduce, cudadevice.MaxThreadsPerBlock(), nBlockSize);

		//cudaDeviceSynchronize();

		// update host data with device data
		o3DMatrix.UpdateHost();

		elapsedTimeGPU = (clock() - startTimeGPU);

		error = cudaEventRecord(eventStop, NULL);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		error = cudaEventSynchronize(eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		float msecTotal = 0.0f;
		error = cudaEventElapsedTime(&msecTotal, eventStart, eventStop);

		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));

			exit(EXIT_FAILURE);
		}

		float msecPerMatrixMul = msecTotal;
		//double flopsPerMatrixMul = 3.0 * (double)nRows * (double)nColumns * (double)nPlanes;
		double flopsPerMatrixMul = (double)nRows * (double)nColumns * (double)nPlanes;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

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

		printf("\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops, WorkgroupSize = %u threads/block\n",
			gigaFlops,
			msecPerMatrixMul,
			flopsPerMatrixMul,
			(nRows * nColumns * nPlanes / nBlockSize));

		//printf("\nElapsed time GPU = %.3f msec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC / 1000);
		//printf("\nElapsed time CPU = %.3f msec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC / 1000);
		printf("\nElapsed time GPU = %.3f sec\n", (double)elapsedTimeGPU / CLOCKS_PER_SEC);
		printf("\nElapsed time CPU = %.3f sec\n", (double)elapsedTimeCPU / CLOCKS_PER_SEC);
	} // 3D matrix

	return 0;
}
