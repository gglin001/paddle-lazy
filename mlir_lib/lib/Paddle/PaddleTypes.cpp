#include "Paddle/PaddleTypes.h"
#include "Paddle/PaddleDialect.h"
#include "Paddle/PaddleOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::paddle;
using namespace mlir::paddle::Paddle;

namespace {

static LogicalResult
verifyTensorType(function_ref<InFlightDiagnostic()> emitError,
                 Optional<ArrayRef<int64_t>> optionalSizes,
                 Type optionalDtype) {
  //   if (optionalDtype && !isValidPaddleDtype(optionalDtype)) {
  //     emitError() << "invalid dtype " << optionalDtype
  //                 << " for !paddle.tensor type";
  //     return failure();
  //   }
  return success();
}

Type parseTensorType(MLIRContext *context, AsmParser &parser,
                     GetTensorTypeFn getTensorType) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseOptionalLess())
    return getTensorType(context,
                         /*optionalSizes=*/None,
                         /*optionalDtype=*/Type());
  bool hasSizes;
  SmallVector<int64_t> sizes;
  if (succeeded(parser.parseOptionalStar())) {
    // Unranked.
    hasSizes = false;
  } else {
    // Parse list of sizes.
    hasSizes = true;
    if (parser.parseLSquare())
      return Type();
    for (bool first = true;; first = false) {
      if (!first) {
        if (failed(parser.parseOptionalComma())) {
          break;
        }
      }
      if (succeeded(parser.parseOptionalQuestion())) {
        sizes.push_back(-1);
        continue;
      }
      int64_t size;
      auto optionalInt = parser.parseOptionalInteger(size);
      if (optionalInt.has_value()) {
        if (failed(*optionalInt))
          return Type();
        sizes.push_back(size);
        continue;
      }
      break;
    }
    if (parser.parseRSquare()) {
      return Type();
    }
  }
  if (parser.parseComma())
    return Type();
  Type optionalDtype;
  if (succeeded(parser.parseOptionalKeyword("unk"))) {
    // Unknown dtype.
  } else {
    // Known dtype.
    if (parser.parseType(optionalDtype))
      return Type();
  }
  if (parser.parseGreater())
    return Type();
  Optional<ArrayRef<int64_t>> optionalSizes;
  if (hasSizes)
    optionalSizes.emplace(sizes);

  if (failed(verifyTensorType([&]() { return parser.emitError(startLoc); },
                              optionalSizes, optionalDtype)))
    return Type();

  return getTensorType(context, optionalSizes, optionalDtype);
}

static void printTensorType(AsmPrinter &printer,
                            Optional<ArrayRef<int64_t>> optionalSizes,
                            Type optionalDtype) {
  if (!optionalSizes && !optionalDtype)
    return;
  printer << "<";
  if (optionalSizes) {
    printer << "[";
    for (auto it : llvm::enumerate(*optionalSizes)) {
      if (it.index() > 0)
        printer << ",";
      if (it.value() < 0)
        printer << "?";
      else
        printer << it.value();
    }
    printer << "]";
  } else {
    printer << "*";
  }
  printer << ",";
  if (optionalDtype)
    printer.printType(optionalDtype);
  else
    printer << "unk";
  printer << ">";
}

} // namespace

// NonValueTensorType
Type NonValueTensorType::parse(::mlir::AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  return parseTensorType(
      context, parser,
      [](MLIRContext *context, Optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType) {
        return NonValueTensorType::get(context, optionalSizes, optionalType);
      });
}

void NonValueTensorType::print(::mlir::AsmPrinter &printer) const {
  printTensorType(printer, getOptionalSizes(), getOptionalDtype());
}

LogicalResult NonValueTensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::Optional<::llvm::ArrayRef<int64_t>> optionalSizes,
    ::mlir::Type optionalDtype) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype);
}
