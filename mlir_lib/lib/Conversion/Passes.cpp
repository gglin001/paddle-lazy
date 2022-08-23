#include "paddle-mlir/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"

#include "paddle-mlir/Conversion/PaddleToTosa/PaddleToTosa.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "paddle-mlir/Conversion/Passes.h.inc"
} // namespace

void mlir::paddle::registerConversionPasses() {
  ::registerConversionPasses();
  //
}
