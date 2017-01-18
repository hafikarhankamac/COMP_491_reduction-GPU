#ifndef ReductionMatrixLib_matrix2D3D_H
#define ReductionMatrixLib_matrix2D3D_H

#include <cassert>

#include "matrixBase.h"
#include "memoryMatrix.h"

namespace ReductionMatrixLib {

	template <class Type> class matrix2D : public matrixBase2D<Type> {
		
		private:
			memoryMatrix<Type> memMatrix;

			size_t IndexofElement(size_t row, size_t column, bool rowmajor) const {
				
				assert(row < this->rows && column < this->columns);

				return ((rowmajor) ? (row * this->columns + column) : (column * this->rows + row));
			}

		public:
			matrix2D() : matrixBase2D<Type>(memMatrix) {

			}

			matrix2D(size_t rows, size_t columns) : matrixBase2D<Type>(memMatrix) {
				
				this->Resize(rows, columns);
			}

			Type &operator()(size_t row, size_t column, bool rowmajor = true) {

				return this->Pointer()[IndexofElement(row, column, rowmajor)];
			}

			Type operator()(size_t row, size_t column, bool rowmajor = true) const {

				return this->Pointer()[IndexofElement(row, column, rowmajor)];
			}
	};

	template <class Type> class matrix3D : public matrixBase3D<Type> {
		
		private:
			memoryMatrix<Type> memMatrix;

			size_t IndexofElement(size_t row, size_t column, size_t plane, bool rowmajor) const {
				
				assert(row < this->rows && column < this->columns && plane < this->planes);

				return ((rowmajor) ? (plane * this->rows * this->columns + row * this->columns + column) : (plane * this->columns * this->rows + column * this->rows + row));
			}

		public:
			matrix3D() : matrixBase3D<Type>(memMatrix) {
				
			}

			matrix3D(size_t rows, size_t columns, size_t planes) : matrixBase3D<Type>(memMatrix) {
				
				this->Resize(rows, columns, planes);
			}

			Type &operator()(size_t row, size_t column, size_t plane, bool rowmajor = true) {
				
				return this->Pointer()[IndexofElement(row, column, plane, rowmajor)];
			}

			Type operator()(size_t row, size_t column, size_t plane, bool rowmajor = true) const {
				
				return this->Pointer()[IndexofElement(row, column, plane, rowmajor)];
			}
	};
}
#endif
