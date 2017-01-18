#ifndef ReductionMatrixLib_memoryMatrix_H
#define ReductionMatrixLib_memoryMatrix_H

#include "memoryBase.h"

namespace ReductionMatrixLib {

	template <class Type> class memoryMatrix : public memoryBase<Type> {

		public:
			virtual void Alloc(size_t size) {

				if (size > 0) {
					this->data = (Type *)malloc(size * sizeof(Type));
					this->size = (this->data) ? size : 0;
				} else {
					this->Reset();
				}
			}

			virtual void Dispose() {

				if (this->size > 0) delete [] this->data;

				this->Reset();
			}

			~memoryMatrix() {

				this->Dispose();
			}
	};
}
#endif
