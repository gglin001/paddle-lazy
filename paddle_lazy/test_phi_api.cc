#include <glog/logging.h>
#include <paddle/extension.h>
#include <paddle/phi/extension.h>

int main() {
  auto a = paddle::full({3, 4}, 2.0);
  auto b = paddle::full({4, 5}, 3.0);
  auto out = paddle::matmul(a, b);
  LOG(ERROR) << out.place() << "|" << out.dtype() << "|"
             << "[" << out.dims() << "]";
}
