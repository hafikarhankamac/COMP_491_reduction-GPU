#ifndef ReductionMatrixLib_MemoryManager_H
#define ReductionMatrixLib_MemoryManager_H

#include <cstddef>

namespace ReductionMatrixLib {

	//! Memory manager base class
	template <class Type> class MemoryManager {

		protected:
			Type * data;
			size_t size;

			void Reset() {

				data = nullptr;
				size = 0;
			}

			MemoryManager() {

				Reset();
			}

		public:
			//! Allocates the memory.
			//! \param size the number of elements of type <Type> that the memory will contain.
			virtual void Alloc(size_t size) = 0;

			//! Disposes (frees) the memory.
			virtual void Dispose() = 0;

			//! Copy data from the device
			//! \param data pointer to the data
			virtual void CopyDataFromDevice(Type * data, size_t size) = 0;

			//! Copy data from the host
			//! \param data pointer to the data
			virtual void CopyDataFromHost(Type * data, size_t size) = 0;

			//! Gets the number of elements (of a generic type) that the memory can contain.
			//! You can use this function to check if the array was effectively allocated.
			//! \return the number of elements hold in memory.
			size_t Size() const {

				return size;
			}

			//! Gets the number of bytes allocated.
			//! You can use this function to check if the array was effectively allocated.
			//! \return the number of of bytes allocated.
			size_t SizeInBytes() const {

				return size * sizeof(Type);
			}

			//! Gets a pointer to the elements (of a generic type) in the memory.
			//! \attention Use with caution
			//! \return a pointer to the elements in the memory
			Type * Pointer() const {

				return data;
			}

			//! Resizes the memory without preserving its data.
			//! \param size the new number of elements (of a generic type) that the memory will contain.
			//! \return the new number of elements in memory.
			size_t ResizeWithoutPreservingData(size_t size) {

				if (size != this->size) {
					Dispose();
					Alloc(size);
				}

				return this->size;
			}

			//! Releases its own resources and obtains ownership of another memory manager resources.
			//! The other manager will no longer have any memory associated.
			//! In other words, it moves the memory from one memory manager to another.
			//! \param other allocator containing the memory resources.
			void TransferOwnerShipFrom(MemoryManager<Type> & other) {

				if (this != &other) {
					Dispose();

					data = other.data;
					size = other.size;

					other.Reset();
				}
			}
	};
}
#endif
