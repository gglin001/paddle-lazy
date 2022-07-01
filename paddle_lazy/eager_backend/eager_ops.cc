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

#include "glog/logging.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

using namespace paddle;  // NOLINT

phi::DeviceContext *GetDeviceContextByBackend(phi::Backend backend) {
  auto &pool = paddle::experimental::DeviceContextPool::Instance();
  return pool.GetMutable(phi::TransToPhiPlace(backend));
}

void dense_copy(DenseTensor *src,
                const Place &place,
                bool blocking,
                DenseTensor *dst) {
  // paddle::framework::TensorCopy(*src, place, src);
  auto *dev_ctx =
      GetDeviceContextByBackend(phi::TransToPhiBackend(src->place()));
  phi::Copy(*dev_ctx, *src, place, false, src);
}

void dense_abs(DenseTensor *x, DenseTensor *out) {
  // use cpu
  Backend kernel_backend = phi::Backend::CPU;
  DataLayout kernel_layout = x->layout();
  DataType kernel_data_type = x->dtype();

  VLOG(6) << "abs API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  const auto &kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "abs", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "abs API kernel: " << kernel;

  auto *dev_ctx = GetDeviceContextByBackend(kernel_backend);

  using kernel_signature = void (*)(
      const phi::DeviceContext &, const phi::DenseTensor &, phi::DenseTensor *);
  auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    // paddle::platform::RecordEvent kernel_record_event("abs compute",
    // paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *x, out);
  }
}

void dense_conv2d(const DenseTensor *input,
                  const DenseTensor *filter,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings,
                  const std::string &paddding_algorithm,
                  int groups,
                  const std::vector<int> &dilations,
                  const std::string &data_format,
                  bool use_addto,
                  int workspace_size_MB,
                  bool exhaustive_search,
                  DenseTensor *out) {
  // use cpu
  Backend kernel_backend = phi::Backend::CPU;
  DataLayout kernel_layout = input->layout();
  DataType kernel_data_type = input->dtype();

  VLOG(6) << "conv2d API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto &kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv2d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "conv2d API kernel: " << kernel;

  auto *dev_ctx = GetDeviceContextByBackend(kernel_backend);

  using kernel_signature = void (*)(const phi::DeviceContext &,
                                    const phi::DenseTensor &,
                                    const phi::DenseTensor &,
                                    const std::vector<int> &,
                                    const std::vector<int> &,
                                    const std::string &,
                                    int,
                                    const std::vector<int> &,
                                    const std::string &,
                                    bool,
                                    int,
                                    bool,
                                    phi::DenseTensor *);
  auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  {
    (*kernel_fn)(*dev_ctx,
                 *input,
                 *filter,
                 strides,
                 paddings,
                 paddding_algorithm,
                 groups,
                 dilations,
                 data_format,
                 use_addto,
                 workspace_size_MB,
                 exhaustive_search,
                 out);
  }
}

void dense_pool2d(const DenseTensor *input,
                  const std::vector<int> &kernel_size,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string &data_format,
                  const std::string &pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string &padding_algorithm,
                  DenseTensor *out) {
  // use cpu
  Backend kernel_backend = phi::Backend::CPU;
  DataLayout kernel_layout = input->layout();
  DataType kernel_data_type = input->dtype();

  VLOG(6) << "pool2d API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  const auto &kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pool2d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  VLOG(6) << "pool2d API kernel: " << kernel;

  auto *dev_ctx = GetDeviceContextByBackend(kernel_backend);

  using kernel_signature = void (*)(const phi::DeviceContext &,
                                    const phi::DenseTensor &,
                                    const std::vector<int> &,
                                    const std::vector<int> &,
                                    const std::vector<int> &,
                                    bool,
                                    bool,
                                    const std::string &,
                                    const std::string &,
                                    bool,
                                    bool,
                                    const std::string &,
                                    phi::DenseTensor *);
  auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    // paddle::platform::RecordEvent kernel_record_event("pool2d compute",
    // paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx,
                 *input,
                 kernel_size,
                 strides,
                 paddings,
                 ceil_mode,
                 exclusive,
                 data_format,
                 pooling_type,
                 global_pooling,
                 adaptive,
                 padding_algorithm,
                 out);
  }
}

}  // namespace phi
