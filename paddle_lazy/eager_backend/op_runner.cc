#include "paddle_lazy/eager_backend/op_runner.h"

#include <glog/logging.h>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle_lazy/eager_backend/eager_ops.h"
#include "paddle_lazy/eager_backend/ops_map.h"
#include "paddle_lazy/lazy_nodes.h"

namespace phi {

void OpRunner::Run(phi::LazyNodePtr node) {
  LOG(ERROR) << "----- lazy running " << node->op_type;
  // TODO(alleng) reduce tensor copy

  for (auto in : node->ins) {
    // LOG(ERROR) << "dense_copy for in: " << in->GetDenseTensor();
    dense_copy(in->GetDenseTensor(), CPUPlace(), false, in->GetDenseTensor());
  }
  // LOG(ERROR) << "fin dense_copy";

  if (GetDenseMap()->count(node->op_type) != 0) {
    GetDenseMap()->at(node->op_type)(node);
  } else if (node->op_type == "assign") {
    auto lazy_node = std::make_shared<AddLazyNode>();
    lazy_node->ins.push_back(node->outs.front());
    lazy_node->ins.push_back(node->ins.front());

    //
    lazy_node->outs.push_back(node->outs.front());
    GetDenseMap()->at("add")(lazy_node);

    // auto out = std::make_shared<phi::DenseTensor>();
    // out->ResizeAndAllocate(node->ins.front()->GetDenseTensor()->dims());
    // auto lazy_out = std::make_shared<LazyTensor>(out.get());
    // lazy_node->outs.push_back(lazy_out);
    // // auto out = paddle::experimental::Tensor();
    // GetDenseMap()->at("add")(lazy_node);
    // // dense_add(const DenseTensor* x, const DenseTensor* y, DenseTensor*
    // out);

    // dense_copy(
    //     // node->ins.front()->GetDenseTensor(),
    //     out.get(),
    //     //  IPUPlace(),
    //     node->ins.front()->GetDenseTensor()->place(),
    //     //  node->outs.front()->GetDenseTensor()->place(),
    //     false,
    //     node->outs.front()->GetDenseTensor());
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("not supported op: %s", node->op_type));
  }
}

void OpRunner::ToCpu(phi::LazyNodePtr node) {
  for (auto t : node->ins) {
    if (!(t->GetDenseTensor()->place().GetType() == phi::AllocationType::CPU) &&
        t->GetDenseTensor()->initialized() &&
        t->GetDenseTensor()->capacity() > 0) {
      dense_copy(t->GetDenseTensor(), CPUPlace(), false, t->GetDenseTensor());
    }
  }

  for (auto t : node->outs) {
    if (!(t->GetDenseTensor()->place().GetType() == phi::AllocationType::CPU) &&
        t->GetDenseTensor()->initialized() &&
        t->GetDenseTensor()->capacity() > 0) {
      dense_copy(t->GetDenseTensor(), CPUPlace(), false, t->GetDenseTensor());
    }
  }
}

void OpRunner::ToIpu(phi::LazyNodePtr node) {
  for (auto t : node->ins) {
    if (!(t->GetDenseTensor()->place().GetType() == phi::AllocationType::IPU)) {
      dense_copy(t->GetDenseTensor(), IPUPlace(), false, t->GetDenseTensor());
    }
  }

  for (auto t : node->outs) {
    if (!(t->GetDenseTensor()->place().GetType() == phi::AllocationType::IPU)) {
      dense_copy(t->GetDenseTensor(), IPUPlace(), false, t->GetDenseTensor());
    }
  }
}

}  // namespace phi
