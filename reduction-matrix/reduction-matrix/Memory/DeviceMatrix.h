#ifndef ReductionMatrixLib_DeviceMatrix_H
#define ReductionMatrixLib_DeviceMatrix_H

#include <cassert>
#include <cublas.h>

#include "../CUDA/reduction_definitions.h"
#include "HostMatrix.h"
#include "DeviceMemoryManager.h"

namespace ReductionMatrixLib {

	//! Create a matrix of any type, on the device, that automatically manages the memory used to hold its elements
	template <class Type> class DeviceMatrix : public BaseMatrix<Type> {
		
		private:
			DeviceMemoryManager<Type> deviceMem;

		public:
			//! Constructs an empty matrix
			//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
			DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {}

			//! Constructs a matrix with a given number of rows and columns
			//! \param rows the number of rows
			//! \param columns the number of columns
			//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
			DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<Type>(deviceMem, storingOrder) {
				
				this->ResizeWithoutPreservingData(rows, columns);
			}

			//! Constructs a matrix identical to the other
			//! \param other another matrix
			DeviceMatrix(const DeviceMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
				
				this->AssignDeviceMatrix(other);
			}

			//! Constructs a matrix identical to an host matrix
			//! \param other host matrix
			DeviceMatrix(const HostMatrix<Type> & other) : BaseMatrix<Type>(deviceMem) {
				
				this->AssignHostMatrix(other);
			}

			//! Transforms this matrix into an matrix identical to an host matrix
			//! \param other host matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			DeviceMatrix<Type> & operator = (const HostMatrix<Type> & other) {
				
				this->AssignHostMatrix(other);
				
				return *this;
			}

			//! Transforms this matrix into an matrix identical to the other
			//! \param other other matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			DeviceMatrix<Type> & operator = (const DeviceMatrix<Type> & other) {
				
				this->AssignDeviceMatrix(other);
				
				return *this;
			}

			//! Constructs a matrix using the elements of a device temporary matrix (rvalue)
			//! \param temporaryMatrix temporary device matrix containing the elements
			DeviceMatrix(DeviceMatrix<Type> && temporaryMatrix) : BaseMatrix<Type>(deviceMem) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
			}

			//! Replaces this matrix using a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix
			//! \return a reference to this matrix
			DeviceMatrix<Type> & operator = (DeviceMatrix<Type> && temporaryMatrix) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
				
				return *this;
			}

			//! Gets the transposed of the matrix
			//! \return the transposed of the matrix
			//! \attention The returned matrix does not use the same method (row-major or column-major) for storing information as this matrix.
			//! \sa ReplaceByTranspose, IsRowMajor
			DeviceMatrix<Type> Transpose() {
				
				HostMatrix<Type> transpose(*this);
				transpose.ReplaceByTranspose();

				return transpose;
			}
	};

	//! Create a cudafloat matrix, on the device, that automatically manages the memory used to hold its elements
	template <> class DeviceMatrix<cudafloat> : public BaseMatrix<cudafloat> {
		
		private:
			DeviceMemoryManager<cudafloat> deviceMem;

		public:
			//! Constructs an empty matrix
			//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
			DeviceMatrix(StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {}

			//! Constructs a matrix with a given number of rows and columns
			//! \param rows the number of rows
			//! \param columns the number of columns
			//! \param storingOrder defines if the matrix uses the row-major or column-major order to store the information
			DeviceMatrix(size_t rows, size_t columns, StoringOrder storingOrder = RowMajor) : BaseMatrix<cudafloat>(deviceMem, storingOrder) {
				
				this->ResizeWithoutPreservingData(rows, columns);
			}

			//! Constructs a matrix identical to another
			//! \param other another matrix
			DeviceMatrix(const DeviceMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
				
				this->AssignDeviceMatrix(other);
			}

			//! Constructs a matrix identical to an host matrix
			//! \param other host matrix
			DeviceMatrix(const HostMatrix<cudafloat> & other) : BaseMatrix<cudafloat>(deviceMem) {
				
				this->AssignHostMatrix(other);
			}

			//! Transforms this matrix into an matrix identical to an host matrix
			//! \param other host matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			DeviceMatrix<cudafloat> & operator = (const HostMatrix<cudafloat> & other) {
				
				this->AssignHostMatrix(other);
				
				return *this;
			}

			//! Transforms this matrix into an matrix identical to the other
			//! \param other other matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			DeviceMatrix<cudafloat> & operator = (const DeviceMatrix<cudafloat> & other) {
				
				this->AssignDeviceMatrix(other);
				
				return *this;
			}

			//! Constructs a matrix using the elements of a device temporary matrix (rvalue)
			//! \param temporaryMatrix temporary device matrix containing the elements
			DeviceMatrix(DeviceMatrix<cudafloat> && temporaryMatrix) : BaseMatrix<cudafloat>(deviceMem) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
			}

			//! Replaces this matrix using a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix
			//! \return a reference to this matrix
			DeviceMatrix<cudafloat> & operator = (DeviceMatrix<cudafloat> && temporaryMatrix) {
				
				this->TransferOwnerShipFrom(temporaryMatrix);
				
				return *this;
			}

			//! Gets the transposed of the matrix
			//! \return the transposed of the matrix
			//! \attention The returned matrix does not use the same method (row-major or column-major) for storing information as this matrix.
			//! \sa ReplaceByTranspose, IsRowMajor
			DeviceMatrix<cudafloat> Transpose() {
				
				HostMatrix<cudafloat> transpose(*this);
				transpose.ReplaceByTranspose();

				return transpose;
			}
	};
}
#endif
