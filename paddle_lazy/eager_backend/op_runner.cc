#include "paddle_lazy/eager_backend/op_runner.h"

#include "paddle_lazy/eager_backend/eager_ops.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {

void OpRunner::Run(phi::LazyNodePtr node) {
  if (node->op_type == "abs") {
    dense_abs(node->ins.front()->GetDenseTensor(),
              node->outs.front()->GetDenseTensor());
  } else if (node->op_type == "conv2d") {
    auto conv2d_node = static_cast<Conv2dLazyNode *>(node.get());
    dense_conv2d(node->ins[0]->GetDenseTensor(),
                 node->ins[1]->GetDenseTensor(),
                 conv2d_node->strides,
                 conv2d_node->dilations_t,
                 conv2d_node->padding_algorithm,
                 conv2d_node->groups,
                 conv2d_node->dilations_t,
                 conv2d_node->data_format,
                 conv2d_node->use_addto,
                 conv2d_node->workspace_size_MB,
                 conv2d_node->exhaustive_search,
                 conv2d_node->outs.front()->GetDenseTensor());
  } else if (node->op_type == "pool2d") {
    auto pool2d_node = static_cast<Pool2dLazyNode *>(node.get());
    dense_pool2d(node->ins[0]->GetDenseTensor(),
                 pool2d_node->kernel_size,
                 pool2d_node->strides,
                 pool2d_node->paddings,
                 pool2d_node->ceil_mode,
                 pool2d_node->exclusive,
                 pool2d_node->data_format,
                 pool2d_node->pooling_type,
                 pool2d_node->global_pooling,
                 pool2d_node->adaptive,
                 pool2d_node->padding_algorithm,
                 pool2d_node->outs.front()->GetDenseTensor());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("not suported op"));
  }
}

}  // namespace phi
