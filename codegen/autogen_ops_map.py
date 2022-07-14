import argparse

import yaml
from base_api import BaseAPI


def gen_cc(cc, cc1):
    return f"""
#pragma once

#include <map>
#include "paddle_lazy/eager_backend/eager_ops.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {{

{cc}

static void init_func() {{

GetDenseMap()->insert(
{{

{cc1}

}});

}};

static int dummy = (init_func(), 0);

}}  // namespace phi

"""


def generate_api(ccs, ccs1, args):
    cc = ''.join(ccs)
    cc1 = ''.join(ccs1)
    cc = gen_cc(cc, cc1)

    with open(args.api_source_path, 'w') as f:
        f.write(cc)


class API(BaseAPI):

    def __init__(self, api_yaml) -> None:
        super().__init__(api_yaml)
        self.api_node = f"{self.api.capitalize()}LazyNode"
        self.api_kernel = f"{self.api.capitalize()}Kernel"
        self.lambda_name = f"lambda_{self.api}"
        self.dense_name = f"dense_{self.api}"

    def cc(self):
        api_code = f"""
auto lambda_{self.api} = [](LazyNodePtr base_node) {{
  auto node = static_cast<{self.api_node}*>(base_node.get());
  return {self.dense_name}({self.get_dense_args()});
}};
"""

        # {"abs", lambda_abs},
        dense_reg = f"""{{"{self.api}", {self.lambda_name}}},
"""
        return api_code, dense_reg

    def get_dense_args(self):
        code = []

        for idx, _ in enumerate(self.inputs['names']):
            code.append(f"node->ins{[idx]}->GetDenseTensor()")

        for idx, _ in enumerate(self.outputs['names']):
            code.append(f"node->outs{[idx]}->GetDenseTensor()")

        for name in self.attrs['names']:
            code.append(f"node->{name}")

        code = ', '.join(code)
        return code


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path',
                        help='path to api yaml file',
                        nargs='+',
                        default=f'codegen/test_api.yaml')
    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle_lazy/eager_backend/ops_map.h')
    args = parser.parse_args()
    # args.api_yaml_path = list(args.api_yaml_path)[0]

    with open(args.api_yaml_path, 'r') as f:
        yaml_list = yaml.load(f, Loader=yaml.FullLoader)

    ccs = []
    ccs1 = []
    for api_yaml in yaml_list:
        api = API(api_yaml)
        cc, cc1 = api.cc()
        ccs.append(cc)
        ccs1.append(cc1)
    generate_api(ccs, ccs1, args)


if __name__ == '__main__':
    main()
