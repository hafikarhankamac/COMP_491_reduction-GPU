#ifndef ReductionMatrixLib_HostMatrix3D_H
#define ReductionMatrixLib_HostMatrix3D_H

#include <cassert>
#include "BaseMatrix3D.h"
#include "HostMemoryManager.h"

namespace ReductionMatrixLib {

	template <class Type> class DeviceMatrix3D;

	//! Create a 3D matrix of any type, on the host, that automatically manages the memory used to hold its elements
	template <class Type> class HostMatrix3D : public BaseMatrix3D<Type> {
		
		private:
			HostMemoryManager<Type> hostMem;

			size_t Index(size_t x, size_t y, size_t z) const {
				
				assert(x < this->dimX && y < this->dimY && z < this->dimZ);

				return z * (this->dimY * this->dimX) + y * this->dimX + x;
			}

		public:
			//! Constructs an empty matrix
			//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
			HostMatrix3D() : BaseMatrix3D<Type>(hostMem) {}

			//! Constructs a matrix with the given dimensions
			//! \param dimX x matrix dimension
			//! \param dimY y matrix dimension
			//! \param dimZ z matrix dimension
			HostMatrix3D(size_t dimX, size_t dimY, size_t dimZ) : BaseMatrix3D<Type>(hostMem) {
				
				this->ResizeWithoutPreservingData(dimX, dimY, dimZ);
			}

			//! Constructs a matrix identical to the other
			//! \param other another matrix
			HostMatrix3D(const HostMatrix3D<Type> & other) : BaseMatrix3D<Type>(hostMem) {
				
				this->AssignHostMatrix(other);
			}

			//! Constructs a matrix identical to a device matrix
			//! \param other device matrix
			HostMatrix3D(const DeviceMatrix3D<Type> & other) : BaseMatrix3D<Type>(hostMem) {
				
				this->AssignDeviceMatrix(other);
			}

			//! Transforms this matrix into an matrix identical to the other
			//! \param other other matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becames the same of the other matrix.
			//! \sa IsRowMajor
			HostMatrix3D<Type> & operator = (const HostMatrix3D<Type> & other) {
				
				this->AssignHostMatrix(other);
				
				return *this;
			}

			//! Transforms this matrix into an matrix identical a device matrix
			//! \param other device matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becames the same of the other matrix.
			//! \sa IsRowMajor
			HostMatrix3D<Type> & operator = (const DeviceMatrix3D<Type> & other) {
				
				this->AssignDeviceMatrix(other);
				
				return *this;
			}

			//! Constructs a matrix using the elements of a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix containing the elements
			HostMatrix3D(HostMatrix3D<Type> && temporaryMatrix) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
			}

			//! Replaces this matrix using a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix
			//! \return a reference to this matrix
			HostMatrix3D<Type> & operator = (HostMatrix3D<Type> && temporaryMatrix) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
				
				return *this;
			}

			//! Gets a reference to an element of the matrix
			//! \param x x index
			//! \param y y index
			//! \param z z index
			//! \return a reference to an element desired, based on the index specified
			Type & operator()(size_t x, size_t y, size_t z) {
				
				return this->Pointer()[Index(x, y, z)];
			}

			//! Gets an element of the matrix
			//! \param x x index
			//! \param y y index
			//! \param z z index
			//! \return the element desired, based on the index specified
			Type operator()(size_t x, size_t y, size_t z) const {
				
				return this->Pointer()[Index(x, y, z)];
			}
	};
}
#endif
