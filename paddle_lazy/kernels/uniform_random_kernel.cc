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

template <typename T>
inline void UniformRealDistribution(T* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T, typename Context>
void UniformRandomRawKernel(const Context& dev_ctx,
                            const phi::IntArray& shape,
                            phi::DataType dtype,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            phi::DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // 1. CPU implement
  phi::DenseTensor cpu_out;
  phi::DenseTensorMeta cpu_out_meta = {out->dtype(), out->dims()};
  cpu_out.set_meta(cpu_out_meta);
  T* cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  UniformRealDistribution<T>(cpu_data, size, min, max, engine);
  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        phi::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      cpu_data[pos] = diag_val;
    }
  }

  // 2. Copy to ipu
  dev_ctx.template Alloc<T>(out);
  phi::Copy(dev_ctx, cpu_out, out->place(), false, out);
}

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const phi::IntArray& shape,
                         phi::DataType dtype,
                         float min,
                         float max,
                         int seed,
                         phi::DenseTensor* out) {
  LOG(ERROR) << "----------- UniformRandomKernel IPU -----------";
  UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_raw,
                   IPU,
                   ALL_LAYOUT,
                   phi::UniformRandomRawKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(
    uniform_random, IPU, ALL_LAYOUT, phi::UniformRandomKernel, float, double) {}
