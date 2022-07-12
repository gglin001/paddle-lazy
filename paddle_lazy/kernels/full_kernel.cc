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

#include <paddle/phi/kernels/full_kernel.h>

#include <paddle/phi/backends/cpu/cpu_context.h>
#include <paddle/phi/core/kernel_registry.h>

namespace phi {

template <typename T, typename Context, typename VType>
void FullValue(const Context& dev_ctx, DenseTensor* tensor, VType val) {
  auto t = dev_ctx.template Alloc<T>(tensor);
  for (auto i = 0; i < tensor->numel(); ++i) {
    t[i] = val;
  }
}

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  FullValue<T>(dev_ctx, out, val.to<T>());
}

}  // namespace phi

PD_REGISTER_KERNEL(full,
                   IPU,
                   ALL_LAYOUT,
                   phi::FullKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
