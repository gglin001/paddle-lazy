#ifndef C43C289A_2E89_4A09_8E84_423C4570A18A
#define C43C289A_2E89_4A09_8E84_423C4570A18A

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace paddle {

#define GEN_PASS_CLASSES
#include "paddle-mlir/Conversion/Passes.h.inc"

} // namespace paddle
} // namespace mlir

#endif // C43C289A_2E89_4A09_8E84_423C4570A18A
