#include "mlir/IR/DialectImplementation.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;
