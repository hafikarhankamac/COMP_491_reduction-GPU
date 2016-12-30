#ifndef ReductionMatrixLib_DeviceMemoryManager_H
#define ReductionMatrixLib_DeviceMemoryManager_H

#include <cuda_runtime.h>
#include "MemoryManager.h"

namespace ReductionMatrixLib {

	//! Device (GPU) memory manager class
	template <class Type> class DeviceMemoryManager : public MemoryManager<Type> {

		public:
			virtual void Alloc(size_t size) {

				if (size > 0 && cudaMalloc((void **) &(this->data), size * sizeof(Type)) == cudaSuccess) {
					this->size = size;
				} else {
					this->Reset();
				}
			}

			virtual void Dispose() {

				if (this->size > 0) cudaFree(this->data);
				this->Reset();
			}

			virtual void CopyDataFromDevice(Type * data, size_t size) {

				this->ResizeWithoutPreservingData(size);

				if (this->size > 0) {
					cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyDeviceToDevice);
				}
			}

			virtual void CopyDataFromHost(Type * data, size_t size) {

				this->ResizeWithoutPreservingData(size);

				if (this->size > 0) {
					cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyHostToDevice);
				}
			}

			~DeviceMemoryManager() {

				Dispose();
			}
	};
}
#endif
