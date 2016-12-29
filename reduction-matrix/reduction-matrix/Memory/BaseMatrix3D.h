#ifndef ReductionMatrixLib_BaseMatrix3D_H
#define ReductionMatrixLib_BaseMatrix3D_H

#include "MemoryManager.h"

namespace ReductionMatrixLib {

	template <class Type> class CudaMatrix3D;

	//! Base class for HostMatrix3D and DeviceMatrix3D classes (3D Matrix base class)
	template <class Type> class BaseMatrix3D {
		
		friend class CudaMatrix3D<Type>;

		protected:
			size_t dimX;
			size_t dimY;
			size_t dimZ;

		private:
			MemoryManager<Type> * mem;

			void CompleteAssign(const BaseMatrix3D<Type> & other) {
				
				if (mem->Size() == other.Elements()) {
					this->dimX = other.dimX;
					this->dimY = other.dimY;
					this->dimZ = other.dimZ;
				}
			}

		public:
			//! Resizes the 3D matrix without preserving its data
			//! \param xDim the new X dimension size of the matrix
			//! \param yDim the new Y dimension size of the matrix
			//! \param zDim the new Z dimension size of the matrix
			//! \return the number of elements of the 3D matrix after being resized.
			size_t ResizeWithoutPreservingData(size_t dimX, size_t dimY, size_t dimZ) {
				
				size_t newElements = dimX * dimY * dimZ;

				if (mem->ResizeWithoutPreservingData(newElements) == newElements) {
					this->dimX = dimX;
					this->dimY = dimY;
					this->dimZ = dimZ;
				} else {
					this->dimX = 0;
					this->dimY = 0;
					this->dimZ = 0;
				}

				return Elements();
			}

		protected:
			BaseMatrix3D(MemoryManager<Type> & mem) {
				
				this->mem = &mem;
				dimX = dimY = dimZ = 0;
			}

			void AssignHostMatrix(const BaseMatrix3D<Type> & other) {
				
				mem->CopyDataFromHost(other.Pointer(), other.Elements());
				CompleteAssign(other);
			}

			void AssignDeviceMatrix(const BaseMatrix3D<Type> & other) {
				
				mem->CopyDataFromDevice(other.Pointer(), other.Elements());
				CompleteAssign(other);
			}

		public:
			//! Disposes the matrix.
			void Dispose() {
				
				mem->Dispose();
				dimX = dimY = dimZ = 0;
			}

			//! Gets the X dimension size of the 3D matrix
			//! \return the number of elements in the X dimension
			size_t DimX() const {
				
				return dimX;
			}

			//! Gets the Y dimension size of the 3D matrix
			//! \return the number of elements in the Y dimension
			size_t DimY() const {
				
				return dimY;
			}

			//! Gets the Z dimension size of the 3D matrix
			//! \return the number of elements in the Z dimension
			size_t DimZ() const {
				
				return dimZ;
			}

			//! Gets a pointer to the 3D matrix data
			//! \attention Use with caution.
			//! \return a pointer to the 3D matrix data
			Type * Pointer() const {
				
				return mem->Pointer();
			}

			//! Gets the number of elements contained in the 3D matrix
			//! \return the number of elements contained in the 3D matrix
			size_t Elements() const {
				
				return mem->Size();
			}

			void TransferOwnerShipFrom(BaseMatrix3D<Type> & other) {
				
				if (this != &other) {
					mem->TransferOwnerShipFrom(*(other.mem));

					dimX = other.dimX;
					dimY = other.dimY;
					dimZ = other.dimZ;

					other.dimX = 0;
					other.dimY = 0;
					other.dimZ = 0;
				}
			}
	};
}
#endif
