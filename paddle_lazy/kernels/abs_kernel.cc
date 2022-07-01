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

#include "paddle/phi/kernels/abs_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/ipu/ipu_context.h"
#include "paddle/phi/backends/ipu/lazy_backend.h"
#include "paddle/phi/backends/ipu/lazy_nodes.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/lazy_tensor.h"

#include "../lazy_allocator.h"

namespace phi {

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  LOG(ERROR) << "----------- AbsKernel IPU -----------";
  LOG(ERROR) << "x.capacity(): " << x.capacity();
  LOG(ERROR) << out->meta().dims.to_str();

  // fake Alloc
  // out->ResetHolder(std::make_shared<phi::Allocation>(nullptr, 0,
  // IPUPlace()));
  out->AllocateFrom(LazyAllocator::Instance(), out->dtype());
  LOG(ERROR) << "fin AllocateFrom";

  auto lazy_x = std::make_shared<LazyTensor>(x);
  auto lazy_out = std::make_shared<LazyTensor>(out);
  auto lazy_node = std::make_shared<AbsLazyNode>();
  lazy_node->ins.push_back(lazy_x);
  lazy_node->outs.push_back(lazy_out);
  // register lazy_node to a global structor
  LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);

  // 这里要和 torch 存在差别, 新的 LazyTensor 与 paddle::experimental::Tensor 和
  // phi::TensorBase 不存在继承关系.
  // 应当是 LazyTensor HAS a (shared) pointer to phi::TensorBase(DenseTensor)
  // LazyNode 用来构建 LazyTensor 的联系和记录op信息(in, out, param)
  // TODO(alleng) 需要解决的问题是如何 在 DenseTensor(TensorBase) 里面记录
  // LazyNode 信息, 一种方法是在创建 LazyNode 后记录到全局的(单例模式)记录中,
  // 后续根据这个记录进行构图.

  // input DenseTensor -> LazyTensor
  // create LazyAbsNode and cache this node
  // do shape inference, get outputs shapes(try to use infermeta/)
  // create output LazyTensors
  // get output DenseTensor from output LazyTensors
  // return output DenseTensors
}

}  // namespace phi

PD_REGISTER_KERNEL(abs,
                   IPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
