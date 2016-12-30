#ifndef ReductionMatrixLib_reduction_init_h
#define ReductionMatrixLib_reduction_init_h

#include <cuda_runtime.h>
#include <iostream>

#include "reduction_definitions.h"

using namespace std;

class CudaDevice {

	private:
		cudaDeviceProp deviceProperties;
		bool deviceSuportsCuda;
		int device;

		void ShowProperty(const char * name, size_t value) {

			cout << name << value << endl;
		}

	public:
		CudaDevice() {

			deviceSuportsCuda = false;

			int numberDevices;

			if (cudaGetDeviceCount(&numberDevices) != cudaSuccess) return;

			for (device = 0; device < numberDevices; device++) {
				if (cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess && deviceProperties.major >= 1) {
					if (cudaSetDevice(device) == cudaSuccess) {
						deviceSuportsCuda = true;

						return;
					}
				}
			}
		}

		bool SupportsCuda() {

			return deviceSuportsCuda;
		}

		int MaxThreadsPerBlock() {

			return deviceProperties.maxThreadsPerBlock;
		}

		int MaxThreadsDimX() {

			return deviceProperties.maxThreadsDim[0];
		}

		int MaxThreadsDimY() {

			return deviceProperties.maxThreadsDim[1];
		}

		int MaxThreadsDimZ() {

			return deviceProperties.maxThreadsDim[2];
		}

		int MaxGridSizeX() {

			return deviceProperties.maxGridSize[0];
		}

		int MaxGridSizeY() {

			return deviceProperties.maxGridSize[1];
		}

		int MaxGridSizeZ() {

			return deviceProperties.maxGridSize[2];
		}

		int OptimalBlockSize() {

			return ((deviceProperties.major < 2) ? 16 : 32);
		}

		void ShowInfo() {

			ShowProperty("Device                                               : ", device);

			cout << "Name                                                 : " << deviceProperties.name << " [" << (deviceProperties.clockRate / 1000) << "Mhz - supports CUDA " << deviceProperties.major << "." << deviceProperties.minor << "]" << endl;

			ShowProperty("Number of processors                                 : ", deviceProperties.multiProcessorCount);
			ShowProperty("Global memory                                        : ", deviceProperties.totalGlobalMem);
			ShowProperty("Max threads per block                                : ", deviceProperties.maxThreadsPerBlock);

			cout << "Max threads of a block for each dimension (x, y, z)  : (" << deviceProperties.maxThreadsDim[0] << ", " << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << ")" << endl;
			cout << "Max block size of a grid for each dimension (x, y, z): (" << deviceProperties.maxGridSize[0] << ", " << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << ")" << endl;

			ShowProperty("Warp size                                            : ", deviceProperties.warpSize);
		}
};
#endif
