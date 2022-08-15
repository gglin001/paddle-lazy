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
