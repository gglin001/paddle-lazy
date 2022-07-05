import re


class BaseAPI:

    def __init__(self, api_yaml) -> None:
        self.api_yaml = api_yaml
        self.api = self.api_yaml['api']
        self.inputs, self.attrs, self.outputs, self.optional_vars = self.parse_args(
        )

        # ignore invoke
        # ignore infer_meta
        # ignore inplace
        # ignore kernel select
        # ignore data_transform

    def parse_args(self):
        optional_vars = []
        if 'optional' in self.api_yaml:
            optional_vars = [
                item.strip() for item in self.api_yaml['optional'].split(',')
            ]
        inputs, attrs = self.parse_input_and_attr(self.api,
                                                  self.api_yaml['args'],
                                                  optional_vars)
        output_type_list, output_names, out_size_expr = self.parse_output(
            self.api, self.api_yaml['output'])
        return inputs, attrs, {
            'names': output_names,
            'types': output_type_list,
            'out_size_expr': out_size_expr
        }, optional_vars

    def parse_input_and_attr(self, api_name, args_config, optional_vars=[]):
        inputs = {'names': [], 'input_info': {}}
        attrs = {'names': [], 'attr_info': {}}
        args_str = args_config.strip()
        assert args_str.startswith('(') and args_str.endswith(')'), \
            f"Args declaration should start with '(' and end with ')', please check the args of {api_name} in yaml."
        args_str = args_str[1:-1]
        args_list = args_str.split(',')
        input_types_map = {
            # 'Tensor': 'const Tensor&',
            'Tensor': 'const DenseTensor*',
            'Tensor[]': 'const std::vector<Tensor>&'
        }
        attr_types_map = {
            'IntArray': 'const IntArray&',
            'Scalar': 'const Scalar&',
            'Scalar(int)': 'const Scalar&',
            'Scalar(int64_t)': 'const Scalar&',
            'Scalar(float)': 'const Scalar&',
            'Scalar(dobule)': 'const Scalar&',
            'int': 'int',
            'int32_t': 'int32_t',
            'int64_t': 'int64_t',
            'long': 'long',
            'size_t': 'size_t',
            'float': 'float',
            'double': 'double',
            'bool': 'bool',
            'str': 'const std::string&',
            'Place': 'const Place&',
            'DataLayout': 'DataLayout',
            'DataType': 'DataType',
            'int64_t[]': 'const std::vector<int64_t>&',
            'int[]': 'const std::vector<int>&'
        }
        optional_types_trans = {
            'Tensor': 'const paddle::optional<Tensor>&',
            'Tensor[]': 'const paddle::optional<std::vector<Tensor>>&',
            'int': 'paddle::optional<int>',
            'int32_t': 'paddle::optional<int32_t>',
            'int64_t': 'paddle::optional<int64_t>',
            'float': 'paddle::optional<float>',
            'double': 'paddle::optional<double>',
            'bool': 'paddle::optional<bool>',
            'Place': 'paddle::optional<const Place&>',
            'DataLayout': 'paddle::optional<DataLayout>',
            'DataType': 'paddle::optional<DataType>'
        }

        for item in args_list:
            item = item.strip()
            type_and_name = item.split(' ')
            # match the input tensor
            has_input = False
            for in_type_symbol, in_type in input_types_map.items():
                if type_and_name[0] == in_type_symbol:
                    input_name = type_and_name[1].strip()
                    assert len(input_name) > 0, \
                        f"The input tensor name should not be empty. Please check the args of {api_name} in yaml."
                    assert len(attrs['names']) == 0, \
                        f"The input Tensor should appear before attributes. please check the position of {api_name}:input({input_name}) in yaml"

                    if input_name in optional_vars:
                        in_type = optional_types_trans[in_type_symbol]

                    inputs['names'].append(input_name)
                    inputs['input_info'][input_name] = in_type
                    has_input = True
                    break
            if has_input:
                continue

            # match the attribute
            for attr_type_symbol, attr_type in attr_types_map.items():
                if type_and_name[0] == attr_type_symbol:
                    attr_name = item[len(attr_type_symbol):].strip()
                    assert len(attr_name) > 0, \
                        f"The attribute name should not be empty. Please check the args of {api_name} in yaml."
                    default_value = None
                    if '=' in attr_name:
                        attr_infos = attr_name.split('=')
                        attr_name = attr_infos[0].strip()
                        default_value = attr_infos[1].strip()

                    if attr_name in optional_vars:
                        attr_type = optional_types_trans[attr_type_symbol]

                    default_value_str = "" if default_value is None else '=' + default_value
                    attrs['names'].append(attr_name)
                    attrs['attr_info'][attr_name] = (attr_type, default_value)
                    break

        return inputs, attrs

    def parse_output(self, api_name, output_config):

        def parse_output_item(output_item):
            output_type_map = {
                'Tensor': 'Tensor',
                'Tensor[]': 'std::vector<Tensor>'
            }
            result = re.search(
                r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*(?P<name>\([a-zA-Z0-9_@]+\))?\s*(?P<expr>\{[^\}]+\})?",
                output_item)
            assert result is not None, f"{api_name} : the output config parse error."
            out_type = result.group('out_type')
            assert out_type in output_type_map, \
                f"{api_name} : Output type error: the output type only support Tensor and Tensor[], \
                  but now is {out_type}."

            out_name = 'out' if result.group('name') is None else result.group(
                'name')[1:-1]
            out_size_expr = None if result.group(
                'expr') is None else result.group('expr')[1:-1]
            return output_type_map[out_type], out_name, out_size_expr

        temp_list = output_config.split(',')

        if len(temp_list) == 1:
            out_type, out_name, size_expr = parse_output_item(temp_list[0])
            return [out_type], [out_name], [size_expr]
        else:
            out_type_list = []
            out_name_list = []
            out_size_expr_list = []
            for output_item in temp_list:
                out_type, out_name, size_expr = parse_output_item(output_item)
                out_type_list.append(out_type)
                out_name_list.append(out_name)
                out_size_expr_list.append(size_expr)

            return out_type_list, out_name_list, out_size_expr_list

    def get_input_tensor_args(self, inplace_flag=False):
        input_args = []
        for name in self.inputs['names']:
            input_args.append(self.inputs['input_info'][name] + ' ' + name)
        return input_args

    def get_output_tensor_args(self):
        output_args = []
        out_type_map = {
            "Tensor": "DenseTensor*",
        }
        for name, dtype in zip(self.outputs['names'], self.outputs['types']):
            dtype = out_type_map[dtype]
            output_args.append(dtype + ' ' + name)
        return output_args

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

    def gene_kernel_select(self) -> str:
        # x is `DenseTensor*`
        x = self.inputs['names'][0]
        kernel_key_item = f"""
  Backend kernel_backend = Backend::CPU;
  DataLayout kernel_layout = {x}->layout();
  DataType kernel_data_type = {x}->dtype();
"""
        return kernel_key_item

    def gen_kernel_code(self, kernel_name, code_indent='', inplace_flag=False):
        kernel_args, kernel_signature = self.get_kernel_args()
        outputs_args = ','.join(self.outputs['names'])
        return f"""
{self.gene_kernel_select()}

{code_indent}  VLOG(6) << "{self.api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
{code_indent}  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{kernel_name}", {{kernel_backend, kernel_layout, kernel_data_type}});
{code_indent}  VLOG(6) << "{kernel_name} kernel: " << kernel;

{code_indent}  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  {{
{code_indent}    (*kernel_fn)({kernel_args}, {outputs_args});
{code_indent}  }}
"""
