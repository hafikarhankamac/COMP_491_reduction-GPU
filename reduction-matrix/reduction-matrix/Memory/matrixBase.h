#ifndef ReductionMatrix_matrixBase_H
#define ReductionMatrix_matrixBase_H

#include "memoryBase.h"

namespace ReductionMatrixLib {

	template <class Type> class matrixBase2D {
		
		protected:
			size_t rows;
			size_t columns;

		private:
			memoryBase<Type> *memBase;

		public:
			size_t Resize(size_t rows, size_t columns) {
				
				size_t newElements = rows * columns;

				if (this->memBase->Resize(newElements) == newElements) {
					this->rows = rows;
					this->columns = columns;
				} else {
					this->rows = 0;
					this->columns = 0;
				}

				return Elements();
			}

		protected:
			matrixBase2D(memoryBase<Type> & memBase) {
				
				this->memBase = &memBase;
				
				this->rows = 0;
				this->columns = 0;
			}

		public:
			void Dispose() {
				
				this->memBase->Dispose();
				
				this->rows = this->columns = 0;
			}

			size_t Rows() const {
				
				return this->rows;
			}

			size_t Columns() const {
				
				return this->columns;
			}

			Type * Pointer() const {
				
				return this->memBase->Pointer();
			}

			size_t Elements() const {
				
				return this->memBase->Size();
			}

			void ReplacebyTranspose() {
				
				size_t newRows = this->columns;

				this->columns = this->rows;
				this->rows = newRows;
			}
	};

	template <class Type> class matrixBase3D {
		
		protected:
			size_t rows;
			size_t columns;
			size_t planes;

		private:
			memoryBase<Type> *memBase;

		public:
			size_t Resize(size_t rows, size_t columns, size_t planes) {
				
				size_t newElements = rows * columns * planes;

				if (this->memBase->Resize(newElements) == newElements) {
					this->rows = rows;
					this->columns = columns;
					this->planes = planes;
				} else {
					this->rows = 0;
					this->columns = 0;
					this->planes = 0;
				}

				return Elements();
			}

		protected:
			matrixBase3D(memoryBase<Type> &memBase) {
				
				this->memBase = &memBase;
				
				this->rows = this->columns = this->planes = 0;
			}

		public:
			void Dispose() {
				
				this->memBase->Dispose();
				
				this->rows = this->columns = this->planes = 0;
			}

			size_t Rows() const {
				
				return this->rows;
			}

			size_t Columns() const {
				
				return this->columns;
			}

			size_t Planes() const {
				
				return this->planes;
			}

			Type * Pointer() const {
				
				return this->memBase->Pointer();
			}

			size_t Elements() const {
				
				return this->memBase->Size();
			}
	};
}
#endif
