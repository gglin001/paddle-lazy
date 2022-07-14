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

#include <paddle/phi/kernels/abs_grad_kernel.h>

#include <paddle/phi/core/kernel_registry.h>
#include <paddle/phi/include/kernels.h>

#include "paddle_lazy/lazy_allocator.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_tensor.h"

namespace phi {

template <typename T, typename Context>
void AbsGradKernel(const Context& ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  LOG(ERROR) << "----------- AbsGradKernel IPU -----------";

  dx->AllocateFrom(LazyAllocator::Instance(), dx->dtype());
  auto lazy_node = std::make_shared<AbsGradLazyNode>();
  auto lazy_x = std::make_shared<LazyTensor>(x);
  lazy_node->ins.push_back(lazy_x);
  auto lazy_dout = std::make_shared<LazyTensor>(dout);
  lazy_node->ins.push_back(lazy_dout);
  auto lazy_dx = std::make_shared<LazyTensor>(dx);
  lazy_node->outs.push_back(lazy_dx);
  LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);
}

}  // namespace phi

PD_REGISTER_KERNEL(abs_grad,
                   IPU,
                   ALL_LAYOUT,
                   phi::AbsGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
