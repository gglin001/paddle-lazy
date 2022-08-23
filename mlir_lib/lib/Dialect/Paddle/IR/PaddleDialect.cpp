#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleTypes.h"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.cpp.inc"

// Tablegen Type Definitions
#define GET_TYPEDEF_CLASSES
#include "paddle-mlir/Dialect/Paddle/IR/PaddleTypes.cpp.inc"

// Dialect initialize method.
void PaddleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "paddle-mlir/Dialect/Paddle/IR/PaddleTypes.cpp.inc"
      >();
}
