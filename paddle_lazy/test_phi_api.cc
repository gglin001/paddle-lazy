#include <glog/logging.h>
#include <paddle/extension.h>
#include <paddle/phi/api/include/api.h>
#include <paddle/phi/extension.h>
#include "paddle/phi/api/ext/tensor_compat.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

#include "paddle_lazy/lazy_backend.h"

using DataType = paddle::experimental::DataType;

void func_0() {
  auto a = paddle::full({3, 4}, 2.0);
  auto b = paddle::full({4, 5}, 3.0);
  auto out = paddle::matmul(a, b);
  LOG(ERROR) << out.place() << "|" << out.dtype() << "|"
             << "[" << out.dims() << "]";
}

void func_1() {
  auto a = paddle::full({3, 4}, 2.0, DataType::FLOAT32, phi::IPUPlace());
  auto out = paddle::relu(a);
  LOG(ERROR) << out.place() << "|" << out.dtype() << "|"
             << "[" << out.dims() << "]";
}

void func_2() {
  auto a = paddle::full({3, 4}, 2.0, DataType::FLOAT32, phi::IPUPlace());
  auto b = a.copy_to(phi::IPUPlace(), false);
  auto out = paddle::sin(b);
  phi::LazyBackend::GetInstance()->Sync();
}

int main() {
  // func_0();
  // func_1();
  func_2();
}
