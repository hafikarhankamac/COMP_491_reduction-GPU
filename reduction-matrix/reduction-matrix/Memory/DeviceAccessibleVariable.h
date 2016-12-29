﻿#ifndef ReductionMatrixLib_DeviceAccessibleVariable_H
#define ReductionMatrixLib_DeviceAccessibleVariable_H

namespace ReductionMatrixLib {

	//! Represents a variable residing in memory that is page-locked and accessible to the device.
	template <class Type> class DeviceAccessibleVariable {

		private:
			Type * value;

		public:
			//! Constructor
			DeviceAccessibleVariable() {
		
				cudaMallocHost((void**) &value, sizeof(Type));
			}

			//! Constructor
			//! \param initialValue Initial value
			DeviceAccessibleVariable(const Type initialValue) {

				cudaMallocHost((void**) &value, sizeof(Type));
				*value = initialValue;
			}

			//! Destructor
			~DeviceAccessibleVariable() {

				cudaFreeHost(value);
			}

			//! Gets a reference to the variable
			//! \return a reference to the variable
			Type & Value() {

				return *value;
			}

			//! Gets a pointer to the variable
			//! \return a pointer to the variable
			Type * Pointer() {

				return value;
			}

			//! Updates the variable value from a device memory variable
			//! \param deviceValue a pointer to the variable on the device
			void UpdateValue(Type * deviceValue) {

				cudaMemcpy(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost);
			}

			//! Asynchronously updates the variable value from a device memory variable
			//! \param deviceValue a pointer to the variable on the device
			//! \param stream The CUDA stream used to transfer the data
			void UpdateValueAsync(Type * deviceValue, cudaStream_t stream) {

				cudaMemcpyAsync(value, deviceValue, sizeof(Type), cudaMemcpyDeviceToHost, stream);
			}
	};
}
#endif
