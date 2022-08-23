#include <paddle/phi/kernels/full_kernel.h>

#include <paddle/phi/backends/cpu/cpu_context.h>
#include <paddle/phi/core/kernel_registry.h>

#include "paddle_lazy/lazy_allocator.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_tensor.h"

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

  if (!out->initialized()) {
    out->AllocateFrom(LazyAllocator::Instance(), out->dtype());
  }
  auto lazy_node = std::make_shared<FullLazyNode>(shape, val, dtype);
  auto lazy_out = std::make_shared<LazyTensor>(out);
  lazy_node->outs.push_back(lazy_out);
  LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);
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
