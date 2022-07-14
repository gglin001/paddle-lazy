#pragma once

#include "paddle_lazy/lazy_backend.h"

namespace phi {

class AbsGradLazyNode : public LazyNode {
 public:
  AbsGradLazyNode() { op_type = "abs_grad"; }
};

}  // namespace phi
