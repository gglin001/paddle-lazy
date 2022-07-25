#pragma once

#include "paddle_lazy/lazy_nodes_autogen.h"

namespace phi {

class AbsGradLazyNode : public LazyNode {
 public:
  AbsGradLazyNode() { op_type = "abs_grad"; }
};

class MeanGradLazyNode : public LazyNode {
 public:
  MeanGradLazyNode(const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all)
      : dims(dims), keep_dim(keep_dim), reduce_all(reduce_all) {
    op_type = "mean_grad";
  }

  std::vector<int64_t> dims;
  bool keep_dim;
  bool reduce_all;
};

}  // namespace phi
