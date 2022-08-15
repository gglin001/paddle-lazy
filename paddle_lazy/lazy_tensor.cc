#include "paddle_lazy/lazy_tensor.h"

namespace phi {

LazyTensor::LazyTensor(const DenseTensor* densetensor)
    : densetensor_(const_cast<DenseTensor*>(densetensor)) {}

LazyTensor::LazyTensor(DenseTensor* densetensor) : densetensor_(densetensor) {}

LazyTensor::LazyTensor(const DenseTensor& densetensor)
    : densetensor_(const_cast<DenseTensor*>(&densetensor)) {}

LazyTensor::LazyTensor() : densetensor_(nullptr) {}

}  // namespace phi
