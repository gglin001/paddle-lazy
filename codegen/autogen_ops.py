import argparse
import os

import yaml
from base_api import BaseAPI


def gen_header(header):
    return f"""
#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace phi {{

{header}

}}  // namespace phi
"""


def gen_cc(cc):
    return f"""
#include "paddle_lazy/eager_backend/autogen_ops.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle_lazy/eager_backend/eager_ops.h"

namespace phi {{

{cc}

}}  // namespace phi
"""


def generate_api(headers, ccs, args):
    header = ''.join(headers)
    header = gen_header(header)

    cc = ''.join(ccs)
    cc = gen_cc(cc)

    with open(args.api_header_path, 'w') as f:
        f.write(header)

    with open(args.api_source_path, 'w') as f:
        f.write(cc)


class API(BaseAPI):

    def header(self):
        api_declaration = f"""
void dense_{self.api}({self.get_declare_args()});
"""
        return api_declaration

    def cc(self):
        api_code = f"""
void dense_{self.api}({self.get_define_args()}) {{
"""
        return api_code + self.gen_kernel_code(self.api) + """
}
"""


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path',
                        help='path to api yaml file',
                        nargs='+',
                        default=f'codegen/test_api.yaml')
    parser.add_argument('--api_header_path',
                        help='output of generated api header code file',
                        default='paddle_lazy/eager_backend/autogen_ops.h')
    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle_lazy/eager_backend/autogen_ops.cc')
    args = parser.parse_args()
    # args.api_yaml_path = list(args.api_yaml_path)[0]

    with open(args.api_yaml_path, 'r') as f:
        yaml_list = yaml.load(f, Loader=yaml.FullLoader)

    headers = []
    ccs = []
    for api_yaml in yaml_list:
        api = API(api_yaml)
        headers.append(api.header())
        ccs.append(api.cc())
    generate_api(headers, ccs, args)


if __name__ == '__main__':
    main()
