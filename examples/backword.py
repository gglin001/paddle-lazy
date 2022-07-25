# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import paddle
import paddle.nn

# fmt: off # yapf: disable
print('\n\n\nload lazy_lib')
import lazy_lib # isort:skip
print('fin load lazy_lib\n\n\n')
# fmt: on # yapf: enable

SEED = 2021
np.random.seed(SEED)
random.seed(SEED)
paddle.seed(SEED)

assert paddle.in_dynamic_mode()
paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})

paddle.set_device('ipu')

data = np.random.uniform(low=-10, high=10, size=[1, 3, 5, 5]).astype(np.float32)
x0 = paddle.to_tensor(data, dtype='float32', stop_gradient=False)

print('\n\nstart op')
print('\n\n')

x = x0

x = paddle.abs(x)
x = paddle.sin(x)

# conv = paddle.nn.Conv2D(3, 3, (3, 3), bias_attr=False)
# x = conv(x)

x = paddle.mean(x)

x.backward(retain_graph=True)

# not work, paddle.grad is a eager method
# x = paddle.grad([x], [x0], retain_graph=True, create_graph=True)

print('start markup')
lazy_lib.lazy.markup()
print('fin markup')

print(x0.grad)
