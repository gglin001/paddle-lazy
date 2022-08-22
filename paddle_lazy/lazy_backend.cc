#include "paddle_lazy/lazy_backend.h"

#include <glog/logging.h>
#include "paddle_lazy/eager_backend/op_runner.h"

#include "import.h"

namespace phi {

LazyBackend *LazyBackend::GetInstance() {
  static LazyBackend instance;
  return &instance;
}

void LazyBackend::Sync() {
  PrettyPrint();
  Compile();
  RunCpu();
}

std::string LazyBackend::PrettyPrint() {
  std::stringstream ss;
  ss << "LazyIr{ \n";
  for (auto node : ir.nodes) {
    ss << LazyNodePrint(node);
  }
  ss << "}\n";

  LOG(ERROR) << "********************************";
  LOG(ERROR) << ss.str();
  LOG(ERROR) << "********************************";
  return ss.str();
}

void LazyBackend::Compile() {
  LOG(ERROR) << "enter LazyBackend::Compile()";
  // convert LazyIr to MLIR graph
  import::run(ir);
}

void LazyBackend::RunCpu() {
  LOG(ERROR) << "enter LazyBackend::Run()";
  auto op_runner = OpRunner();
  for (auto node : ir.nodes) {
    op_runner.Run(node);
  }

  for (auto node : ir.nodes) {
    op_runner.ToIpu(node);
  }
}

void LazyBackend::RunIpu() {
  //
  //
}

std::string LazyNodePrint(const LazyNodePtr node) {
  std::stringstream ss;
  size_t count = 0;
  ss << "\t" << node->op_type << ", (";
  for (auto in : node->ins) {
    if (count > 0) {
      ss << ", ";
    }
    auto t = in->GetDenseTensor();
    ss << DTPrint(t) << " | " << in->GetDenseTensor();
    ++count;
  }
  count = 0;
  ss << ") -> (";
  for (auto out : node->outs) {
    if (count > 0) {
      ss << ", ";
    }
    auto t = out->GetDenseTensor();
    ss << DTPrint(t) << " | " << out->GetDenseTensor();
    ++count;
  }
  ss << ")\n";
  return ss.str();
}

std::string DTPrint(const DenseTensor *t) {
  std::stringstream ss;
  ss << t->place() << "|";
  ss << t->dtype() << "|";
  std::string init_state;
  if (!t->initialized()) {
    ss << "Uninitialized|";
  } else if (t->capacity() == 0) {
    ss << "Empty|";
  } else {
    ss << "Initialized|"
       << "Cap:" << t->capacity() << "|";
  }
  ss << "[" << t->dims() << "]";
  return ss.str();
};

}  // namespace phi
