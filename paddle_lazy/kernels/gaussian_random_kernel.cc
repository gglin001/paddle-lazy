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

#include <paddle/phi/kernels/gaussian_random_kernel.h>

#include <glog/logging.h>
#include <paddle/phi/backends/ipu/ipu_context.h>
#include <paddle/phi/core/kernel_registry.h>
#include <paddle/phi/core/tensor_utils.h>

namespace phi {

template <typename T, typename Context>
void GaussianRandomKernel(const Context& ctx,
                          const IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          DataType dtype,
                          DenseTensor* out) {
  LOG(ERROR) << "----------- GaussianRandomKernel IPU -----------";
  // cpu kernel
  phi::DenseTensor cpu_tensor;
  phi::DenseTensorMeta cpu_meta = {out->dtype(), out->dims()};
  cpu_tensor.set_meta(cpu_meta);
  T* cpu_data = ctx.template HostAlloc<T>(&cpu_tensor);
  std::normal_distribution<T> dist(mean, std);
  int64_t size = out->numel();
  auto gen_ptr = ctx.GetGenerator();
  gen_ptr->SetCurrentSeed(static_cast<int64_t>(seed));
  auto engine = gen_ptr->GetCPUEngine();
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = dist(*engine);
  }

  // copy to ipu
  ctx.template Alloc<T>(out);
  phi::Copy(ctx, cpu_tensor, out->place(), false, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian_random,
                   IPU,
                   ALL_LAYOUT,
                   phi::GaussianRandomKernel,
                   float,
                   double) {}
