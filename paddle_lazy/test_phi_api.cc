#include <glog/logging.h>
#include <paddle/extension.h>
#include <paddle/phi/extension.h>

int main() {
  LOG(ERROR) << "abc is: "
             << "abc";
}
