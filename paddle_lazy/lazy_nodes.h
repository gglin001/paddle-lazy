#pragma once

#include "paddle_lazy/lazy_nodes_autogen.h"

namespace phi {

class AssignLazyNode : public LazyNode {
 public:
  AssignLazyNode() { op_type = "assign"; }
};

}  // namespace phi
