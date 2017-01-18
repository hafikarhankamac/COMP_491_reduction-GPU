#ifndef ReductionMatrixLib_memoryBase_H
#define ReductionMatrixLib_memoryBase_H

namespace ReductionMatrixLib {

	template <class Type> class memoryBase {

		protected:
			Type * data;
			size_t size;

			void Reset() {

				data = 0;
				size = 0;
			}

			memoryBase() {

				Reset();
			}

		public:
			virtual void Alloc(size_t size) = 0;

			virtual void Dispose() = 0;

			size_t Size() const {

				return this->size;
			}

			size_t SizeinBytes() const {

				return this->size * sizeof(Type);
			}

			Type * Pointer() const {

				return this->data;
			}

			size_t Resize(size_t size) {

				if (size != this->size) {
					this->Dispose();
					this->Alloc(size);
				}

				return this->size;
			}
	};
}
#endif
