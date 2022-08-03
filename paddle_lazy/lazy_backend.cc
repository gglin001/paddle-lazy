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

#include <glog/logging.h>
#include "paddle_lazy/eager_backend/op_runner.h"

namespace phi {

LazyBackend *LazyBackend::GetInstance() {
  static LazyBackend instance;
  return &instance;
}

void LazyBackend::Sync() {
  PrettyPrint();
  Compile();
  RunCpu();
  PrettyPrint();
  ir.nodes.clear();
}

std::string LazyBackend::PrettyPrint() {
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
      ss << DTPrint(t) << "|" << in->GetDenseTensor();
      ++count;
    }
    count = 0;
    ss << ") -> (";
    for (auto out : node->outs) {
      if (count > 0) {
        ss << ", ";
      }
      auto t = out->GetDenseTensor();
      ss << DTPrint(t) << "|" << out->GetDenseTensor();
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
  auto op_runner = OpRunner();
  for (auto node : ir.nodes) {
    op_runner.Run(node);
  }

  for (auto node : ir.nodes) {
    op_runner.ToIpu(node);
  }
}

void LazyBackend::RunIpu() {
  //
  //
}

std::string DTPrint(const DenseTensor *t) {
  std::stringstream ss;
  ss << t->place() << "|";
  ss << t->dtype() << "|";
  std::string init_state;
  if (!t->initialized()) {
    ss << "Uninitialized|";
  } else if (t->capacity() == 0) {
    ss << "Empty|";
  } else {
    // ss << t->place() << "|";
    ss << "Initialized|"
       << "Cap:" << t->capacity() << "|";
  }
  ss << "[" << t->dims() << "]";
  return ss.str();
};

}  // namespace phi
