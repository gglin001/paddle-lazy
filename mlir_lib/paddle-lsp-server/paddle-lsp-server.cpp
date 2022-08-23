#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::paddle::Paddle::PaddleDialect>();

  return failed(MlirLspServerMain(argc, argv, registry));
}
