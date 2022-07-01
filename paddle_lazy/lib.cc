#include "lib.h"

#include <glog/logging.h>
#include <paddle/extension.h>
#include <paddle/phi/extension.h>
#include <pybind11/pybind11.h>

void lib() { LOG(ERROR) << "enter a lib"; }

int mul(int i, int j) { return i * j; }

namespace py = pybind11;  // NOLINT

PYBIND11_MODULE(lazy_lib, m) {
  m.def("lib", &lib);

  m.def("mul", &mul);
}
