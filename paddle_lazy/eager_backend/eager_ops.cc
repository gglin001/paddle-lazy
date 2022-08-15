#include "paddle_lazy/eager_backend/eager_ops.h"

#include <glog/logging.h>
#include <paddle/phi/api/include/context_pool.h>
#include <paddle/phi/common/backend.h>
#include <paddle/phi/common/place.h>
#include <paddle/phi/core/dense_tensor.h>
#include <paddle/phi/core/device_context.h>
#include <paddle/phi/core/kernel_factory.h>
#include <paddle/phi/core/tensor_utils.h>

namespace phi {

using namespace paddle;  // NOLINT

std::map<std::string, std::function<void(LazyNodePtr)>>* GetDenseMap() {
  static std::map<std::string, std::function<void(LazyNodePtr)>> dense_map;
  return &dense_map;
}

phi::DeviceContext* GetDeviceContextByBackend(phi::Backend backend) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  return pool.GetMutable(phi::TransToPhiPlace(backend));
}

void dense_copy(DenseTensor* src,
                const Place& place,
                bool blocking,
                DenseTensor* dst) {
  auto* dev_ctx = GetDeviceContextByBackend(phi::TransToPhiBackend(place));
  phi::Copy(*dev_ctx, *src, place, false, src);
}

}  // namespace phi
