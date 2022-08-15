#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Paddle/PaddleDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register paddle passes here.

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::paddle::PaddleDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Paddle optimizer driver\n", registry));
}