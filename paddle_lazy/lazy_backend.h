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

#pragma once

#include "paddle/phi/core/lazy_tensor.h"

namespace phi {

class LazyNode {
 public:
  LazyNode() = default;

  std::string op_type;
  // TODO(alleng) args
  std::vector<LazyTensorPtr> ins;
  std::vector<LazyTensorPtr> outs;
};

using LazyNodePtr = std::shared_ptr<LazyNode>;

class LazyIr {
 public:
  std::vector<LazyNodePtr> nodes;
};

class LazyBackend {
 public:
  static LazyBackend* GetInstance();

 public:
  void Sync();

  std::string PrettyPrint();
  void Compile();
  void RunCpu();
  void RunIpu();

  LazyIr ir;
};

}  // namespace phi
