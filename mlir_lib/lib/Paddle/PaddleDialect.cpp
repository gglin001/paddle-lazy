#include "Paddle/PaddleDialect.h"
#include "Paddle/PaddleOps.h"
#include "Paddle/PaddleTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::paddle;

#include "Paddle/PaddleDialect.cpp.inc"

// Tablegen Type Definitions
#define GET_TYPEDEF_CLASSES
#include "Paddle/PaddleTypes.cpp.inc"

// Dialect initialize method.
void PaddleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Paddle/PaddleOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Paddle/PaddleTypes.cpp.inc"
      >();
}
