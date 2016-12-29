#ifndef ReductionMatrixLib_CUDAMatrix3D_H
#define ReductionMatrixLib_CUDAMatrix3D_H

#include "DeviceMatrix3D.h"

namespace ReductionMatrixLib {

	//! Create a matrix of any type, that automatically manages the memory used to hold its elements (data will be stored both on the host and on the device).
	//! \attention The data on the host might differ from the data on the device. Use UpdateDevice and UpdateHost to synchronize data.
	template <class Type> class CudaMatrix3D {

		private:
			HostMatrix3D<Type> h;
			DeviceMatrix3D<Type> d;

			void TransferOwnerShipFrom(CudaMatrix3D<Type> & other) {
				
				if (this != other) {
					h.TransferOwnerShipFrom(other.h);
					d.TransferOwnerShipFrom(other.d);
				}
			}

		public:
			//! Constructs an empty matrix
			CudaMatrix3D() : h(), d() {}

			//! Constructs a matrix with the given dimensions
			//! \param dimX x matrix dimension
			//! \param dimY y matrix dimension
			//! \param dimZ z matrix dimension
			CudaMatrix3D(size_t dimX, size_t dimY, size_t dimZ) : h(dimX, dimY, dimZ), d(dimX, dimY, dimZ) {}

			//! Constructs a matrix identical to a device matrix
			//! \param other device matrix
			CudaMatrix3D(const DeviceMatrix3D<Type> & other) : h(other), d(other) {}

			//! Constructs a matrix identical to an host matrix
			//! \param other another matrix
			CudaMatrix3D(const HostMatrix3D<Type> & other) : h(other), d(other) {}

			//! Constructs a matrix identical to another matrix
			//! \param other another matrix
			CudaMatrix3D(const CudaMatrix3D<Type> & other) : h(other.h), d(other.d) {}

			//! Transforms this matrix into a matrix identical to another
			//! \param other other matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			CudaMatrix3D<Type> & operator = (const CudaMatrix3D<Type> & other) {
				
				h = other.h;
				d = other.d;

				return *this;
			}

			//! Transforms this matrix into a matrix identical to a device matrix
			//! \param other device matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			CudaMatrix3D<Type> & operator = (const DeviceMatrix3D<Type> & other) {
				
				h = d = other;

				return *this;
			}

			//! Transforms this matrix into an matrix identical to a host matrix
			//! \param other host matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			CudaMatrix3D<Type> & operator = (const HostMatrix3D<Type> & other) {
				
				d = h = other;

				return *this;
			}

			//! Constructs a matrix using the elements of a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix containing the elements
			CudaMatrix3D(CudaMatrix3D<Type> && temporaryMatrix) {
				
				TransferOwnerShipFrom(temporaryMatrix);
			}

			//! Replaces this matrix using a temporary matrix (rvalue)
			//! \param temporaryMatrix temporary matrix
			//! \return a reference to this matrix
			//! \attention The storing order (major-row or major-column) becomes the same of the other matrix.
			//! \sa IsRowMajor
			CudaMatrix3D<Type> & operator = (const CudaMatrix3D<Type> && temporaryMatrix) {
				
				TransferOwnerShipFrom(temporaryMatrix);
				
				return *this;
			}

			//! Updates the device matrix data with the host matrix data
			//! \attention The storing order (major-row or major-column) of the device matrix becomes the same as the host matrix.
			//! \sa IsRowMajor
			void UpdateDevice() {
				
				d = h;
			}

			//! Updates the host matrix data with the device matrix data
			//! \attention The storing order (major-row or major-column) of the host matrix becomes the same as the device matrix.
			//! \sa IsRowMajor
			void UpdateHost() {
				
				h = d;
			}

			//! Gets a pointer to the host matrix data
			//! \attention Use with caution. Special attention should be given to how the matrix information is stored (row-major or column-major).
			//! \return a pointer to the matrix data
			//! \sa IsRowMajor
			Type * HostPointer() const {
				
				return h.Pointer();
			}

			//! Gets a pointer to the device matrix data
			//! \attention Use with caution
			//! \return a pointer to the device matrix data
			Type * DevicePointer() const {
				
				return d.Pointer();
			}

			//! Gets a the device matrix
			//! \attention Use with caution
			//! \return The device matrix
			DeviceMatrix3D<Type> & Get3DDeviceMatrix() {
				
				return d;
			}

			//! Gets a the host matrix
			//! \attention Use with caution
			//! \return The host matrix
			HostMatrix3D<Type> & Get3DHostMatrix() {
				
				return h;
			}

			//! Gets a reference to an element of the host matrix
			//! \param x x index
			//! \param y y index
			//! \param z z index
			//! \return a reference to an element desired, based on the index specified
			Type & operator()(size_t x, size_t y, size_t z) {
				
				return h(x, y, z);
			}

			//! Gets an element of the host matrix
			//! \param x x index
			//! \param y y index
			//! \param z z index
			//! \return the element desired, based on the index specified
			Type operator()(size_t x, size_t y, size_t z) const {
				
				return h(x, y, z);
			}

			//! Gets the X dimension size of the 3D matrix
			//! \return the number of elements in the X dimension
			size_t DimX() const {
				
				return h.DimX();
			}

			//! Gets the Y dimension size of the 3D matrix
			//! \return the number of elements in the Y dimension
			size_t DimY() const {
				
				return h.DimY();
			}

			//! Gets the Z dimension size of the 3D matrix
			//! \return the number of elements in the Z dimension
			size_t DimZ() const {
				
				return h.DimZ();
			}

			//! Gets the number of elements contained in the matrix
			//! \return the number of elements contained in the matrix
			int Elements() const {
				
				return h.Elements();
			}


			//! Resizes the 3D matrix without preserving its data
			//! \param xDim the new X dimension size of the matrix
			//! \param yDim the new Y dimension size of the matrix
			//! \param zDim the new Z dimension size of the matrix
			//! \return the number of elements of the 3D matrix after being resized.
			size_t ResizeWithoutPreservingData(size_t dimX, size_t dimY, size_t dimZ) {
				
				size_t he = h.ResizeWithoutPreservingData(dimX, dimY, dimZ);
				size_t de = d.ResizeWithoutPreservingData(dimX, dimY, dimZ);

				return (he > de) ? de : he;
			}

			//! Disposes the matrix
			void Dispose() {
				
				d.Dispose();
				h.Dispose();
			}

			//! //! Updates the host information and disposes the device matrix
			void DisposeDevice() {
				
				UpdateHost();
				d.Dispose();
			}

			//! Updates the device information and disposes the host matrix
			void DisposeHost() {
				
				UpdateDevice();
				h.Dispose();
			}
	};
}
#endif
