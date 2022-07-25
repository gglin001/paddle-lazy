#include <glog/logging.h>
#include <paddle/extension.h>
#include <paddle/phi/extension.h>
#include <pybind11/pybind11.h>

#include "paddle_lazy/lazy_backend.h"

namespace py = pybind11;  // NOLINT

PYBIND11_MODULE(lazy_lib, m) {
  auto lazy = m.def_submodule("lazy", "Lazy mode");
  lazy.def("markup", []() {
    LOG(ERROR) << "LazyTensor call lazy.markup()";
    phi::LazyBackend::GetInstance()->Sync();
    LOG(ERROR) << "LazyTensor fin lazy.markup()";
  });
  lazy.def("debug", []() { phi::LazyBackend::GetInstance()->PrettyPrint(); });
}
