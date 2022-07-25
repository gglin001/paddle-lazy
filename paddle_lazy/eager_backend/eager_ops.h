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

#include <paddle/phi/api/include/context_pool.h>
#include <paddle/phi/core/dense_tensor.h>

#include "paddle_lazy/eager_backend/eager_ops_autogen.h"
#include "paddle_lazy/lazy_backend.h"

namespace phi {

// singleton
std::map<std::string, std::function<void(LazyNodePtr)>>* GetDenseMap();

phi::DeviceContext* GetDeviceContextByBackend(phi::Backend backend);

void dense_copy(DenseTensor* src,
                const Place& place,
                bool blocking,
                DenseTensor* dst);

}  // namespace phi
