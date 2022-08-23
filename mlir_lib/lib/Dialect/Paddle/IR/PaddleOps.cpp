#include "mlir/IR/OpImplementation.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"

#define GET_OP_CLASSES
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.cpp.inc"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;
