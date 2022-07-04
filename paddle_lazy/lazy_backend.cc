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

#include "paddle_lazy/lazy_backend.h"

#include "glog/logging.h"
#include "paddle_lazy/eager_backend/eager_ops.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {

LazyBackend *LazyBackend::GetInstance() {
  static LazyBackend instance;
  return &instance;
}

void LazyBackend::Sync() {
  PrettyPrint();
  Compile();
  RunCpu();
}

std::string LazyBackend::PrettyPrint() {
  auto tensor_state = [](const DenseTensor *t) -> std::string {
    std::stringstream ss;
    ss << t->place() << "|" << t->dtype() << "|";
    std::string init_state;
    if (!t->initialized()) {
      init_state = "Uninitialized|";
    } else if (t->capacity() == 0) {
      init_state = "Empty|";
    } else {
      init_state = "Initialized|";
    }
    ss << init_state << "[" << t->dims() << "]";
    return ss.str();
  };

  std::stringstream ss;
  ss << "LazyIr{ \n";
  for (auto node : ir.nodes) {
    size_t count = 0;
    ss << "\t" << node->op_type << ", (";
    for (auto in : node->ins) {
      if (count > 0) {
        ss << ", ";
      }
      auto t = in->GetDenseTensor();
      ss << tensor_state(t);
      ++count;
    }
    count = 0;
    ss << ") -> (";
    for (auto out : node->outs) {
      if (count > 0) {
        ss << ", ";
      }
      auto t = out->GetDenseTensor();
      ss << tensor_state(t);
      ++count;
    }
    ss << ")\n";
  }
  ss << "}\n";

  LOG(ERROR) << "********************************";
  LOG(ERROR) << ss.str();
  LOG(ERROR) << "********************************";
  return ss.str();
}

void LazyBackend::Compile() {
  LOG(ERROR) << "enter LazyBackend::Compile()";
  // convert LazyIr to LazyGraph(DAG)
  // LazyGraph should be runable one by one step(op)
}

void LazyBackend::RunCpu() {
  LOG(ERROR) << "enter LazyBackend::Run()";
  for (auto node : ir.nodes) {
    LOG(ERROR) << "----- lazy running " << node->op_type;

    // TODO(alleng) reduce tensor copy
    for (auto node : node->ins) {
      dense_copy(
          node->GetDenseTensor(), CPUPlace(), false, node->GetDenseTensor());
    }

    if (node->op_type == "abs") {
      dense_abs(node->ins.front()->GetDenseTensor(),
                node->outs.front()->GetDenseTensor());
    } else if (node->op_type == "conv2d") {
      auto conv2d_node = static_cast<Conv2dLazyNode *>(node.get());
      dense_conv2d(node->ins[0]->GetDenseTensor(),
                   node->ins[1]->GetDenseTensor(),
                   conv2d_node->strides,
                   conv2d_node->dilations_t,
                   conv2d_node->padding_algorithm,
                   conv2d_node->groups,
                   conv2d_node->dilations_t,
                   conv2d_node->data_format,
                   conv2d_node->use_addto,
                   conv2d_node->workspace_size_MB,
                   conv2d_node->exhaustive_search,
                   conv2d_node->outs.front()->GetDenseTensor());
    } else if (node->op_type == "pool2d") {
      auto pool2d_node = static_cast<Pool2dLazyNode *>(node.get());
      dense_pool2d(node->ins[0]->GetDenseTensor(),
                   pool2d_node->kernel_size,
                   pool2d_node->strides,
                   pool2d_node->paddings,
                   pool2d_node->ceil_mode,
                   pool2d_node->exclusive,
                   pool2d_node->data_format,
                   pool2d_node->pooling_type,
                   pool2d_node->global_pooling,
                   pool2d_node->adaptive,
                   pool2d_node->padding_algorithm,
                   pool2d_node->outs.front()->GetDenseTensor());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("not suported op"));
    }
  }
}

void LazyBackend::RunIpu() {
  //
  //
}

}  // namespace phi
