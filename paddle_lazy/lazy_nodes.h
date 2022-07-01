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

#include "paddle/phi/backends/ipu/lazy_backend.h"

namespace phi {

class AbsLazyNode : public LazyNode {
 public:
  AbsLazyNode() { op_type = "abs"; }
};

class Conv2dLazyNode : public LazyNode {
 public:
  Conv2dLazyNode(const std::vector<int>& strides,
                 const std::vector<int>& paddings_t,
                 const std::string& padding_algorithm,
                 int groups,
                 const std::vector<int>& dilations_t,
                 const std::string& data_format,
                 bool use_addto,
                 int workspace_size_MB,
                 bool exhaustive_search)
      : strides(strides),
        paddings_t(paddings_t),
        padding_algorithm(padding_algorithm),
        groups(groups),
        dilations_t(dilations_t),
        data_format(data_format),
        use_addto(use_addto),
        workspace_size_MB(workspace_size_MB),
        exhaustive_search(exhaustive_search) {
    op_type = "conv2d";
  }

  std::vector<int> strides;
  std::vector<int> paddings_t;
  std::string padding_algorithm;
  int groups;
  std::vector<int> dilations_t;
  std::string data_format;
  bool use_addto;
  int workspace_size_MB;
  bool exhaustive_search;
};

class Pool2dLazyNode : public LazyNode {
 public:
  Pool2dLazyNode(const std::vector<int>& kernel_size,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 bool ceil_mode,
                 bool exclusive,
                 const std::string& data_format,
                 const std::string& pooling_type,
                 bool global_pooling,
                 bool adaptive,
                 const std::string& padding_algorithm)
      : kernel_size(kernel_size),
        strides(strides),
        paddings(paddings),
        ceil_mode(ceil_mode),
        exclusive(exclusive),
        data_format(data_format),
        pooling_type(pooling_type),
        global_pooling(global_pooling),
        adaptive(adaptive),
        padding_algorithm(padding_algorithm) {
    op_type = "pool2d";
  }

  std::vector<int> kernel_size;
  std::vector<int> strides;
  std::vector<int> paddings;
  bool ceil_mode;
  bool exclusive;
  std::string data_format;
  std::string pooling_type;
  bool global_pooling;
  bool adaptive;
  std::string padding_algorithm;
};

}  // namespace phi
