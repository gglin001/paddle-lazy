import argparse

import yaml
from base_api import BaseAPI


def gen_header(header):
    return f"""
#pragma once

#include <tuple>

#include <paddle/phi/api/include/tensor.h>
#include <paddle/phi/common/scalar.h>
#include <paddle/phi/common/int_array.h>
#include <paddle/utils/optional.h>

namespace phi {{

{header}

}}  // namespace phi
"""


def gen_cc(cc):
    return f"""
#include "paddle_lazy/eager_backend/eager_ops_autogen.h"

#include <memory>

#include <glog/logging.h>
#include <paddle/phi/api/include/context_pool.h>
#include <paddle/phi/core/dense_tensor.h>
#include <paddle/phi/core/kernel_registry.h>
#include <paddle/phi/infermeta/binary.h>
#include <paddle/phi/infermeta/multiary.h>
#include <paddle/phi/infermeta/nullary.h>
#include <paddle/phi/infermeta/ternary.h>
#include <paddle/phi/infermeta/unary.h>

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

    def get_declare_args(self, inplace_flag=False):
        declare_args = self.get_input_tensor_args(inplace_flag)
        declare_args.extend(self.get_output_tensor_args())
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            declare_args.append(self.attrs['attr_info'][name][0] + ' ' + name +
                                default_value)
        return ", ".join(declare_args)

    def get_define_args(self, inplace_flag=False):
        define_args = self.get_input_tensor_args(inplace_flag)
        define_args.extend(self.get_output_tensor_args())
        for name in self.attrs['names']:
            define_args.append(self.attrs['attr_info'][name][0] + ' ' + name)
        return ", ".join(define_args)

    def get_kernel_args(self, kernel_tensor_type=None, code_indent=''):
        dense_input_trans_map = {
            # 'const Tensor&':
            'const DenseTensor*':
            'const phi::DenseTensor&',
            'const std::vector<Tensor>&':
            'const std::vector<const phi::DenseTensor*>&',
            'const paddle::optional<Tensor&>':
            'paddle::optional<const phi::DenseTensor&>',
            'const paddle::optional<Tensor>&':
            'const paddle::optional<phi::DenseTensor>&',
            'const paddle::optional<std::vector<Tensor>>&':
            'paddle::optional<const std::vector<phi::DenseTensor>&>'
        }
        dense_out_trans_map = {
            'Tensor': 'phi::DenseTensor*',
            'std::vector<Tensor>': 'std::vector<phi::DenseTensor*>&'
        }
        sr_input_trans_map = {
            'const Tensor&':
            'const phi::SelectedRows&',
            'const paddle::optional<Tensor>&':
            'const paddle::optional<phi::SelectedRows>&'
        }
        sr_out_trans_map = {'Tensor': 'phi::SelectedRows*'}
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const phi::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_param = input_names + attr_names

        kernel_args = ["*dev_ctx"]
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_args.append(param)
                else:
                    if self.inputs['input_info'][param] == "const DenseTensor*":
                        kernel_args.append("*" + param)
                    elif self.inputs['input_info'][
                            param] == "const std::vector<Tensor>&":
                        kernel_args.append(param)
                    else:
                        # do nothing
                        pass
                # input is dense tensor
                if kernel_tensor_type is None or kernel_tensor_type[0][
                        kernel_param.index(param)] == 'dense':
                    kernel_args_type_list.append(
                        dense_input_trans_map[input_infos[param]])
                else:  # input is selected_rows
                    kernel_args_type_list.append(
                        sr_input_trans_map[input_infos[param]])
            elif param in attr_names:
                # set attr for kernel_context
                if 'IntArray' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::IntArray&')
                    param = 'phi::IntArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    param = 'phi::Scalar(' + param + ')'
                else:
                    kernel_args_type_list.append(
                        self.attrs['attr_info'][param][0])
                kernel_args.append(param)
            elif isinstance(param, bool):
                kernel_args.append(str(param).lower())
            else:
                kernel_args.append(str(param))

        for i, out_type in enumerate(self.outputs['types']):
            # output is dense tensor
            if kernel_tensor_type is None or kernel_tensor_type[1][i] == 'dense':
                kernel_args_type_list.append(dense_out_trans_map[out_type])
            else:  # output is selected_rows
                kernel_args_type_list.append(sr_out_trans_map[out_type])
        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"
        kernel_args = ", ".join(kernel_args)

        return kernel_args, kernel_signature

    def gen_kernel_code(self, kernel_name, code_indent='', inplace_flag=False):
        kernel_args, kernel_signature = self.get_kernel_args()
        outputs_args = ','.join(self.outputs['names'])
        return f"""
  Backend kernel_backend = Backend::CPU;
  DataLayout kernel_layout = {self.inputs['names'][0]}->layout();
  DataType kernel_data_type = {self.inputs['names'][0]}->dtype();


{code_indent}  VLOG(6) << "{self.api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
{code_indent}  const auto& kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{kernel_name}", {{kernel_backend, kernel_layout, kernel_data_type}});
{code_indent}  const auto& kernel = kernel_result.kernel;
{code_indent}  VLOG(6) << "{kernel_name} kernel: " << kernel;

{code_indent}  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  {{
{code_indent}    (*kernel_fn)({kernel_args}, {outputs_args});
{code_indent}  }}
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
                        default='paddle_lazy/eager_backend/eager_ops_autogen.h')
    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle_lazy/eager_backend/eager_ops_autogen.cc')
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
