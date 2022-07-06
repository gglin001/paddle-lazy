import argparse

import yaml
from base_api import BaseAPI


def gen_cc(cc):
    return f"""
#include "paddle/phi/include/kernels.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle_lazy/lazy_allocator.h"
#include "paddle_lazy/lazy_backend.h"
#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_tensor.h"

namespace phi {{

{cc}

}}  // namespace phi

#include "paddle_lazy/kernels/register.cc.inc"

"""


def generate_api(ccs, args):
    cc = ''.join(ccs)
    cc = gen_cc(cc)

    with open(args.api_source_path, 'w') as f:
        f.write(cc)


class API(BaseAPI):

    def __init__(self, api_yaml) -> None:
        super().__init__(api_yaml)
        self.api_node = f"{self.api.capitalize()}LazyNode"
        self.api_kernel = f"{self.api.capitalize()}Kernel"

    def cc(self):
        api_code = f"""
template <typename T, typename Context>
void {self.api_kernel}({self.get_init_args()}) {{
"""
        return api_code + self.gen_kernel_code() + """
}
"""

    def get_init_args(self, inplace_flag=False):
        declare_args = ['const Context& dev_ctx']
        declare_args.extend(self.get_input_tensor_args(inplace_flag))
        for name in self.attrs['names']:
            declare_args.append(self.attrs['attr_info'][name][0] + ' ' + name)
        declare_args.extend(self.get_output_tensor_args())
        declare_args = ", ".join(declare_args)
        declare_args = declare_args.replace('const DenseTensor*',
                                            'const DenseTensor&')
        return declare_args

    def gen_kernel_code(self):
        code = []
        code.append(f"""
LOG(ERROR) << "----------- {self.api_kernel} IPU -----------";
""")
        # fake alloc
        for name in self.outputs['names']:
            code.append(
                f"{name}->AllocateFrom(LazyAllocator::Instance(), {name}->dtype());"
            )

        # create LazyNode
        attrs = ', '.join(self.attrs['names'])
        code.append(
            f"auto lazy_node = std::make_shared<{self.api_node}>({attrs});")

        # inputs to LazyTensor
        for name in self.inputs['names']:
            code.append(
                f"auto lazy_{name} = std::make_shared<LazyTensor>({name});")
            code.append(f"lazy_node->ins.push_back(lazy_{name});")

        # outputs to LazyTensor
        for name in self.outputs['names']:
            code.append(
                f"auto lazy_{name} = std::make_shared<LazyTensor>({name});")
            code.append(f"lazy_node->outs.push_back(lazy_{name});")

        # register LazyNode
        code.append(
            f"LazyBackend::GetInstance()->ir.nodes.push_back(lazy_node);")

        code = '\n'.join(code)
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
                        default='paddle_lazy/kernels/kernels.cc')
    args = parser.parse_args()
    # args.api_yaml_path = list(args.api_yaml_path)[0]

    with open(args.api_yaml_path, 'r') as f:
        yaml_list = yaml.load(f, Loader=yaml.FullLoader)

    ccs = []
    for api_yaml in yaml_list:
        api = API(api_yaml)
        ccs.append(api.cc())
    generate_api(ccs, args)


if __name__ == '__main__':
    main()
