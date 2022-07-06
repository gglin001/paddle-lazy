#include "paddle_lazy/eager_backend/op_runner.h"

#include "paddle_lazy/eager_backend/all_ops.h"
#include "paddle_lazy/eager_backend/ops_map.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {

void OpRunner::Run(phi::LazyNodePtr node) {
  // TODO(alleng) reduce tensor copy
  for (auto node : node->ins) {
    dense_copy(
        node->GetDenseTensor(), CPUPlace(), false, node->GetDenseTensor());
  }

  if (dense_map.count(node->op_type) == 0) {
    PADDLE_THROW(
        phi::errors::Unimplemented("not supported op: %s", node->op_type));
  } else {
    dense_map.at(node->op_type)(node);
  }
}

}  // namespace phi
