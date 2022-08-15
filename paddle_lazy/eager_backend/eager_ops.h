#pragma once

#include <paddle/phi/api/include/context_pool.h>
#include <paddle/phi/core/dense_tensor.h>

#include "paddle_lazy/eager_backend/eager_ops_autogen.h"
#include "paddle_lazy/lazy_backend.h"

namespace phi {

// singleton
std::map<std::string, std::function<void(LazyNodePtr)>>* GetDenseMap();

phi::DeviceContext* GetDeviceContextByBackend(phi::Backend backend);

void dense_copy(DenseTensor* src,
                const Place& place,
                bool blocking,
                DenseTensor* dst);

}  // namespace phi
