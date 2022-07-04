#pragma once

#include "paddle_lazy/lazy_backend.h"

namespace phi {

class OpRunner {
 public:
  OpRunner() = default;

  void Run(phi::LazyNodePtr);
};

}  // namespace phi
