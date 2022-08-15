#!/bin/bash

# run at ../

python codegen/autogen_eager_ops.py
python codegen/autogen_kernels.py
python codegen/autogen_lazy_nodes.py
python codegen/autogen_ops_map.py
