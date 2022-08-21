#include "Paddle/PaddleTypes.h"
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

#include "Paddle/PaddleDialect.h"
#include "Paddle/PaddleOps.h"
#include "Paddle/PaddleTypes.h"

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
  mlir::Value one = builder.create<Paddle::ConstantOp>(
      mlir::UnknownLoc::get(&context), ty, oneAttr);
  builder.create<Paddle::ReluOp>(mlir::UnknownLoc::get(&context), one.getType(),
                                 one);

  module->dump();
}
