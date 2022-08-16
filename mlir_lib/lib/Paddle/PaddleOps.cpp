#include "Paddle/PaddleOps.h"
#include "Paddle/PaddleDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Paddle/PaddleOps.cpp.inc"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;
