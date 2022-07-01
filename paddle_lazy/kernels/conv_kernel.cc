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

#include "paddle/phi/kernels/conv_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/ipu/ipu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_tensor.h"

#include "../lazy_allocator.h"

namespace phi {

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter_t,
                const std::vector<int>& strides,
                const std::vector<int>& paddings_t,
                const std::string& padding_algorithm,
                int groups,
                const std::vector<int>& dilations_t,
                const std::string& data_format,
                bool use_addto,
                int workspace_size_MB,
                bool exhaustive_search,
                DenseTensor* output) {
  LOG(ERROR) << "----------- ConvKernel IPU -----------";
  // output->ResetHolder(
  //     std::make_shared<phi::Allocation>(nullptr, 0, IPUPlace()));
  output->AllocateFrom(LazyAllocator::Instance(), output->dtype());

  LOG(ERROR) << "filter_t: " << filter_t.place() << "|" << filter_t.dtype()
             << "|" << filter_t.dims();

  auto lazy_input = std::make_shared<LazyTensor>(input);
  auto lazy_filter_t = std::make_shared<LazyTensor>(filter_t);
  auto lazy_output = std::make_shared<LazyTensor>(output);
  auto lazy_node = std::make_shared<Conv2dLazyNode>(strides,
                                                    paddings_t,
                                                    padding_algorithm,
                                                    groups,
                                                    dilations_t,
                                                    data_format,
                                                    use_addto,
                                                    workspace_size_MB,
                                                    exhaustive_search);
  lazy_node->ins.push_back(lazy_input);
  lazy_node->ins.push_back(lazy_filter_t);
  lazy_node->outs.push_back(lazy_output);
  LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d, IPU, ALL_LAYOUT, phi::ConvKernel, float, double) {}
