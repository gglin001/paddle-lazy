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

#include "paddle/phi/kernels/pool_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/ipu/ipu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle_lazy/lazy_allocator.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_tensor.h"

namespace phi {

template <typename T, typename Context>
void Pool2dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  //
  LOG(ERROR) << "----------- Pool2dKernel IPU -----------";
  // out->ResetHolder(std::make_shared<phi::Allocation>(nullptr, 0,
  // IPUPlace()));
  out->AllocateFrom(LazyAllocator::Instance(), out->dtype());

  auto lazy_input = std::make_shared<LazyTensor>(x);
  auto lazy_output = std::make_shared<LazyTensor>(out);
  auto lazy_node = std::make_shared<Pool2dLazyNode>(kernel_size,
                                                    strides,
                                                    paddings,
                                                    ceil_mode,
                                                    exclusive,
                                                    data_format,
                                                    pooling_type,
                                                    global_pooling,
                                                    adaptive,
                                                    padding_algorithm);
  lazy_node->ins.push_back(lazy_input);
  lazy_node->outs.push_back(lazy_output);
  LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);
}

}  // namespace phi

PD_REGISTER_KERNEL(pool2d, IPU, ALL_LAYOUT, phi::Pool2dKernel, float, double) {}
