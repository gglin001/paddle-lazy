import argparse

import yaml
from base_api import BaseAPI


def gen_header(header):
    return f"""
#pragma once

#include "paddle_lazy/lazy_backend.h"

namespace phi {{

{header}

}}  // namespace phi
"""


def gen_cc(cc):
    return f"""
#include "paddle_lazy/lazy_nodes.h"

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

    def __init__(self, api_yaml) -> None:
        super().__init__(api_yaml)
        self.api_node = f"{self.api.capitalize()}LazyNode"

    def header(self):
        api_declaration = f"""
class {self.api_node} : public LazyNode {{
 public:

{self.api_node}{self.get_init_args()}{{
op_type = "{self.api}";
}}

{self.get_params()}

}};
"""
        return api_declaration

    def cc(self):
        api_code = f"""
"""
        return api_code

    def get_init_args(self):
        declare_args = []
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            declare_args.append(self.attrs['attr_info'][name][0] + ' ' + name +
                                default_value)
        declare_args = ", ".join(declare_args)
        declare_args = f"({declare_args})"

        if len(self.attrs['names']) > 0:
            declare_args = f"{declare_args}:\n"

        init_args = []
        for name in self.attrs['names']:
            init_args.append(f"{name}({name})")
        init_args = ", ".join(init_args)

        return declare_args + init_args

    def get_params(self):
        params = []
        for name in self.attrs['names']:
            param_type = self.attrs['attr_info'][name][0]
            param_type = param_type.replace('&', '')
            param_type = param_type.replace('const', '')
            param_type = param_type.strip()
            params.append(f"{param_type} {name};")
        params = "\n".join(params)
        return params


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path',
                        help='path to api yaml file',
                        nargs='+',
                        default=f'codegen/test_api.yaml')
    parser.add_argument('--api_header_path',
                        help='output of generated api header code file',
                        default='paddle_lazy/lazy_nodes_autogen.h')
    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle_lazy/lazy_nodes_autogen.cc')
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
