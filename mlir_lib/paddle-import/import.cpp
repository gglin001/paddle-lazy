#include "import.h"

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
#include <llvm-16/llvm/Support/raw_ostream.h>

#include "Paddle/PaddleDialect.h"
#include "Paddle/PaddleOps.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;

namespace import {

void run(phi::LazyIr &ir) {
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

  for (auto &node : ir.nodes) {
    auto &node_type = node->op_type;
    if (node_type == "sin") {
      builder.create<Paddle::SinOp>(mlir::UnknownLoc::get(&context),
                                    one.getType(), one);
    } else {
      llvm::errs() << "unsupported op type: " << node_type << "\n";
    }
  }

  llvm::errs() << "module->dump():\n";
  module->dump();
}

} // namespace import
