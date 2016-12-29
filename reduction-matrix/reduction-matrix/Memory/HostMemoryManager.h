#ifndef ReductionMatrixLib_HostMemoryManager_H
#define ReductionMatrixLib_HostMemoryManager_H

#include <cuda_runtime.h>
#include <cstring>
#include <new>

#include "MemoryManager.h"

namespace ReductionMatrixLib {

	//! Host (CPU) memory manager class
	template <class Type> class HostMemoryManager : public MemoryManager<Type> {

		public:
			virtual void Alloc(size_t size) {

				if (size > 0) {
					this->data = new (std::nothrow) Type[size];
					this->size = (this->data != nullptr) ? size : 0;
				} else {
					this->Reset();
				}
			}

			virtual void Dispose() {

				if (this->size > 0) delete [] this->data;

				this->Reset();
			}

			virtual void CopyDataFromDevice(Type * data, size_t size) {

				this->ResizeWithoutPreservingData(size);

				if (this->size > 0) {
					cudaMemcpy(this->data, data, this->SizeInBytes(), cudaMemcpyDeviceToHost);
				}
			}

			virtual void CopyDataFromHost(Type * data, size_t size) {

				this->ResizeWithoutPreservingData(size);

				if (this->size > 0) {
					memcpy(this->data, data, this->SizeInBytes());
				}
			}

			~HostMemoryManager() {

				Dispose();
			}
	};
}
#endif
