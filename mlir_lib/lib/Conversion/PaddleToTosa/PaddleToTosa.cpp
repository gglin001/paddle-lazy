#include "paddle-mlir/Conversion/PaddleToTosa/PaddleToTosa.h"

#include "../PassDetail.h"

namespace mlir {
namespace paddle {

namespace {
class ConvertPaddleToTosa : public PaddleToTosaBase<ConvertPaddleToTosa> {
public:
  void runOnOperation() override {
    //
  }
};
} // namespace

std::unique_ptr<Pass> createPaddleToTosaPass() {
  return std::make_unique<ConvertPaddleToTosa>();
};

} // namespace paddle
} // namespace mlir
