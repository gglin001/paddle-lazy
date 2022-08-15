import os

os.environ["FLAGS_enable_eager_mode"] = "1"

import numpy as np
import paddle
import paddle.fluid
import paddle.fluid.framework
import paddle.nn

# fmt: off # yapf: disable
print('\n\n\nload lazy_lib')
import lazy_lib # isort:skip
print('fin load lazy_lib\n\n\n')
# fmt: on # yapf: enable

paddle.set_device("ipu")
paddle.fluid.framework._disable_legacy_dygraph()

data = np.random.uniform(low=-10, high=10, size=[1, 3, 5, 5]).astype(np.float32)
x = paddle.to_tensor(
    data,
    dtype="float32",
    stop_gradient=False,
    place=paddle.IPUPlace(),
)

print("start op")
print("\n\n")

# call twice
x = paddle.abs(x)
x = paddle.abs(x)

conv = paddle.nn.Conv2D(3, 3, (3, 3), bias_attr=False)
x = conv(x)

pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
x = pool(x)

print("start markup")
lazy_lib.lazy.markup()
print("fin markup")

print(x)
