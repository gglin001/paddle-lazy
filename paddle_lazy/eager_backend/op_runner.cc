#include "paddle_lazy/eager_backend/op_runner.h"

#include <glog/logging.h>
#include "paddle_lazy/eager_backend/eager_ops.h"
#include "paddle_lazy/eager_backend/ops_map.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {

void OpRunner::Run(phi::LazyNodePtr node) {
  LOG(ERROR) << "----- lazy running " << node->op_type;
  // TODO(alleng) reduce tensor copy
  for (auto in : node->ins) {
    dense_copy(in->GetDenseTensor(), CPUPlace(), false, in->GetDenseTensor());
  }

  if (GetDenseMap()->count(node->op_type) != 0) {
    GetDenseMap()->at(node->op_type)(node);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("not supported op: %s", node->op_type));
  }
}

}  // namespace phi
