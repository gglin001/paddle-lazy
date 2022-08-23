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
#include "llvm/Support/raw_ostream.h"

#include <map>

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"

#include "paddle_lazy/lazy_nodes.h"
#include "paddle_lazy/lazy_nodes_autogen.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;
using namespace phi;

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
  std::map<DenseTensor *, mlir::Value> value_map;

  for (auto &node : ir.nodes) {
    auto &node_type = node->op_type;
    if (node_type == "sin") {
      auto n = static_cast<SinLazyNode *>(node.get());
      mlir::Value value_in = value_map[n->ins[0]->GetDenseTensor()];
      builder.create<Paddle::SinOp>(mlir::UnknownLoc::get(&context),
                                    value_in.getType(), value_in);
    } else if (node_type == "full") {

      auto n = static_cast<FullLazyNode *>(node.get());
      auto shape_ = n->shape.GetData();
      ArrayRef<int64_t> shape{shape_};
      // TODO(allen) use `value` and `dtype`
      ElementsAttr oneAttr = builder.getDenseF32ArrayAttr({1.0});
      auto ty = RankedTensorType::get(shape, builder.getF32Type());
      // auto ty = UnrankedTensorType::get(builder.getF32Type());
      mlir::Value full = builder.create<Paddle::ConstantOp>(
          mlir::UnknownLoc::get(&context), ty, oneAttr);
      value_map[n->outs[0]->GetDenseTensor()] = full;
    } else {
      llvm::errs() << "unsupported op type: " << node_type << "\n";
    }
  }

  llvm::errs() << "module->dump():\n";
  module->dump();
}

} // namespace import
