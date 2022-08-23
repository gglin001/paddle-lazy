#include "paddle-mlir/Conversion/PaddleToTosa/PaddleToTosa.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

#include "paddle-mlir/Dialect/Paddle/IR/PaddleDialect.h"
#include "paddle-mlir/Dialect/Paddle/IR/PaddleOps.h"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;

namespace mlir {
namespace paddle {

namespace {

class TanOpLowering : public OpRewritePattern<TanOp> {
public:
  using OpRewritePattern<TanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TanOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::TanhOp>(op, op->getResults().getType(),
                                              op->getOperands());
    return success();
  }
};

class ConvertPaddleToTosa : public PaddleToTosaBase<ConvertPaddleToTosa> {
public:
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect>();
    RewritePatternSet patterns(&getContext());

    target.addIllegalOp<TanOp>();
    patterns.add<TanOpLowering>(&getContext());

    // TODO support TypeConverter
    TypeConverter typeConverter;

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createPaddleToTosaPass() {
  return std::make_unique<ConvertPaddleToTosa>();
};

} // namespace paddle
} // namespace mlir
