#include "Paddle/PaddleTypes.h"
#include "Paddle/PaddleDialect.h"
#include "Paddle/PaddleOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "Paddle/PaddleTypes.cpp.inc"
