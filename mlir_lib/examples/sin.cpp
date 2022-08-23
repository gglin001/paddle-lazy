#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
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

  auto zeroAttr = builder.getZeroAttr(builder.getF32Type());
  mlir::Value zero = builder.create<arith::ConstantOp>(
      mlir::UnknownLoc::get(&context), zeroAttr);
  builder.create<Paddle::SinOp>(mlir::UnknownLoc::get(&context),
                                builder.getF32Type(), zero);

  module->dump();
}
