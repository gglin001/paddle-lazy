#pragma once

#include <paddle/phi/core/dense_tensor.h>

namespace phi {

class LazyTensor {
 public:
  explicit LazyTensor(const DenseTensor* densetensor);
  explicit LazyTensor(DenseTensor* densetensor);
  explicit LazyTensor(const DenseTensor& densetensor);

  LazyTensor();

  DenseTensor* GetDenseTensor() const { return densetensor_; }

 private:
  // replace to TensorBase
  // TODO(alleng) use const
  DenseTensor* densetensor_;
};

using LazyTensorPtr = std::shared_ptr<LazyTensor>;

}  // namespace phi
