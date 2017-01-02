#ifndef ReductionMatrixLib_reduction_init_h
#define ReductionMatrixLib_reduction_init_h

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "reduction_definitions.h"

//using namespace std;

#if CUDART_VERSION < 5000 /* CUDART_VERSION < 5000 */

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T> inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device) {

	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(stderr, "\ncuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n", error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

inline int _ConvertSMVer2Cores(int major, int minor) {

	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
			return nGpuArchCoresPerSM[index].Cores;

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("\nMapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);

	return nGpuArchCoresPerSM[index - 1].Cores;
}

class CudaDevice {

	private:
		int numberDevices;
		int device;

		cudaDeviceProp deviceProperties;

		bool deviceSuportsCuda;

public:
		CudaDevice() {

			deviceSuportsCuda = false;

			cudaError_t errorCuda;

			if ((errorCuda = cudaGetDeviceCount(&numberDevices)) != cudaSuccess) {
				printf("\ncudaGetDeviceCount returned %d -> %s\n", (int)errorCuda, cudaGetErrorString(errorCuda));

				return;
			}

			// Find the first CUDA capable GPU device
			for (device = 0; device < numberDevices; device++) {
				if ((errorCuda = cudaGetDeviceProperties(&deviceProperties, device)) != cudaSuccess) {
					printf("\ncudaGetDeviceProperties returned %d -> %s\n", (int)errorCuda, cudaGetErrorString(errorCuda));

					return;
				}

				if (deviceProperties.major >= 1) {
					if ((errorCuda = cudaSetDevice(device)) != cudaSuccess) {
						printf("\ncudaSetDevice returned %d -> %s\n", (int)errorCuda, cudaGetErrorString(errorCuda));

						return;
					}

					deviceSuportsCuda = true;
						
					return;
				}
			}
		}

		bool SupportsCuda() {

			return deviceSuportsCuda;
		}

		float ProcessingPowerGFlops() {

			unsigned long long compute_perf = 0;

			int sm_per_multiproc = 0;

			if (deviceProperties.computeMode != cudaComputeModeProhibited) {
				if (deviceProperties.major == 9999 && deviceProperties.minor == 9999)
					sm_per_multiproc = 1;
				else
					sm_per_multiproc = _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor);

				compute_perf = (float)(deviceProperties.multiProcessorCount * sm_per_multiproc * deviceProperties.clockRate);
			}

			compute_perf *= 2; // for single presicion

			return (compute_perf / 1024 / 1024);
		}

		unsigned long SharedMemPerBlock() {

			return deviceProperties.sharedMemPerBlock;
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

			int driverVersion = 0, runtimeVersion = 0;

			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			printf("  Device:                                        %d\n", device);
			printf("  Device Name:                                   %s\n", deviceProperties.name);

			printf("  CUDA Driver Version / Runtime Version:         %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProperties.major, deviceProperties.minor);

			char msg[256];
			sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProperties.totalGlobalMem / 1048576.0f, (unsigned long long)deviceProperties.totalGlobalMem);
			printf("%s", msg);

			printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n", deviceProperties.multiProcessorCount, _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor), _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor) * deviceProperties.multiProcessorCount);
			printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProperties.clockRate * 1e-3f, deviceProperties.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000 /* CUDART_VERSION >= 5000 */

			// This is supported in CUDA 5.0 (runtime API device properties)
			printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProperties.memoryClockRate * 1e-3f);
			printf("  Memory Bus Width:                              %d-bit\n", deviceProperties.memoryBusWidth);

			if (deviceProperties.l2CacheSize)
				printf("  L2 Cache Size:                                 %d bytes\n", deviceProperties.l2CacheSize);

#else /* CUDART_VERSION < 5000 */

			// This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
			int memoryClock;

			getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
			
			printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
			
			int memBusWidth;
			
			getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
			
			printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
			
			int L2CacheSize;
			
			getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

			if (L2CacheSize)
				printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);

#endif /* CUDART_VERSION >= 5000 */

			printf("  Maximum Texture Dimension Size (x,y,z):        1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n", deviceProperties.maxTexture1D, deviceProperties.maxTexture2D[0], deviceProperties.maxTexture2D[1], deviceProperties.maxTexture3D[0], deviceProperties.maxTexture3D[1], deviceProperties.maxTexture3D[2]);
			printf("  Maximum Layered 1D Texture Size, (num) layers: 1D=(%d), %d layers\n", deviceProperties.maxTexture1DLayered[0], deviceProperties.maxTexture1DLayered[1]);
			printf("  Maximum Layered 2D Texture Size, (num) layers: 2D=(%d, %d), %d layers\n", deviceProperties.maxTexture2DLayered[0], deviceProperties.maxTexture2DLayered[1], deviceProperties.maxTexture2DLayered[2]);

			printf("  Total amount of constant memory:               %lu bytes\n", deviceProperties.totalConstMem);
			printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProperties.sharedMemPerBlock);
			printf("  Total number of registers available per block: %d\n", deviceProperties.regsPerBlock);
			printf("  Warp size:                                     %d\n", deviceProperties.warpSize);
			printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProperties.maxThreadsPerMultiProcessor);
			printf("  Maximum number of threads per block:           %d\n", deviceProperties.maxThreadsPerBlock);
			printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
			printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
			printf("  Maximum memory pitch:                          %lu bytes\n", deviceProperties.memPitch);
			printf("  Texture alignment:                             %lu bytes\n", deviceProperties.textureAlignment);
			printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProperties.deviceOverlap ? "Yes" : "No"), deviceProperties.asyncEngineCount);
			printf("  Run time limit on kernels:                     %s\n", deviceProperties.kernelExecTimeoutEnabled ? "Yes" : "No");
			printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProperties.integrated ? "Yes" : "No");
			printf("  Support host page-locked memory mapping:       %s\n", deviceProperties.canMapHostMemory ? "Yes" : "No");
			printf("  Alignment requirement for Surfaces:            %s\n", deviceProperties.surfaceAlignment ? "Yes" : "No");
			printf("  Device has ECC support:                        %s\n", deviceProperties.ECCEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

			printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProperties.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");

#endif

			printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProperties.unifiedAddressing ? "Yes" : "No");
			printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProperties.pciDomainID, deviceProperties.pciBusID, deviceProperties.pciDeviceID);

			printf("  Device Processing Power (Single Precision):    %.2f GFlop/s\n", ProcessingPowerGFlops());

			const char *sComputeMode[] = {
				"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
				"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
				"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
				"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
				"Unknown",
				NULL
			};

			printf("  Compute Mode:                                  < %s >\n", sComputeMode[deviceProperties.computeMode]);
		}
};
#endif
