// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  // paddle::framework::TensorCopy(*src, place, src);
  auto* dev_ctx =
      GetDeviceContextByBackend(phi::TransToPhiBackend(src->place()));
  phi::Copy(*dev_ctx, *src, place, false, src);
}

void dense_abs_grad(const DenseTensor* x,
                    const DenseTensor* dout,
                    DenseTensor* out) {
  Backend kernel_backend = Backend::CPU;
  DataLayout kernel_layout = x->layout();
  DataType kernel_data_type = x->dtype();

  VLOG(6) << "abs_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "abs_grad", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "abs_grad kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  using kernel_signature = void (*)(const phi::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  { (*kernel_fn)(*dev_ctx, *x, *dout, out); }
}

}  // namespace phi
