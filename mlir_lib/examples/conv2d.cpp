#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleTypes.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;

int main() {
  mlir::registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<PaddleDialect>();

  MLIRContext context;
  OwningOpRef<mlir::ModuleOp> module =
      ModuleOp::create(mlir::UnknownLoc::get(&context));
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(module->getBodyRegion());

  ElementsAttr oneAttr = builder.getDenseF32ArrayAttr({1.0});
  auto ty = UnrankedTensorType::get(builder.getF32Type());
  mlir::Value x = builder.create<Paddle::ConstantOp>(
      mlir::UnknownLoc::get(&context), ty, oneAttr);
  mlir::Value w = builder.create<Paddle::ConstantOp>(
      mlir::UnknownLoc::get(&context), ty, oneAttr);
  auto strides = builder.getI32ArrayAttr({0, 0});
  auto paddings = builder.getI32ArrayAttr({0, 0});
  auto padding_algorithm = builder.getStringAttr({"EXPLICIT"});
  auto groups = builder.getI32IntegerAttr(1);
  auto dilations = builder.getI32ArrayAttr({0, 0});
  auto data_format = builder.getStringAttr({"NCHW"});
  builder.create<Paddle::Conv2dOp>(mlir::UnknownLoc::get(&context), x.getType(),
                                   x, w, strides, paddings, padding_algorithm,
                                   groups, dilations, data_format);

  module->dump();
}
