#pragma once

#include "paddle_lazy/lazy_tensor.h"

namespace phi {

class LazyNode {
 public:
  LazyNode() = default;

  std::string op_type;
  // TODO(alleng) args
  std::vector<LazyTensorPtr> ins;
  std::vector<LazyTensorPtr> outs;
};

using LazyNodePtr = std::shared_ptr<LazyNode>;

class LazyIr {
 public:
  std::vector<LazyNodePtr> nodes;
};

class LazyBackend {
 public:
  static LazyBackend *GetInstance();

 public:
  void Sync();

  std::string PrettyPrint();
  void Compile();
  void RunCpu();
  void RunIpu();

  LazyIr ir;
};

std::string DTPrint(const DenseTensor *);
std::string LazyNodePrint(const LazyNodePtr);

}  // namespace phi
