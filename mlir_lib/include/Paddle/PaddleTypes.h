#ifndef F109FBEB_EEAF_4D85_8D01_AC401C3AD5F5
#define F109FBEB_EEAF_4D85_8D01_AC401C3AD5F5

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace paddle {
namespace Paddle {

constexpr static int64_t kUnknownSize = -1;

class BaseTensorType : public Type {
public:
  using Type::Type;

  /// Get the raw optional list of sizes.
  ///
  /// It is expected that for many users, `hasSizes`/`getSizes` will be a more
  /// convenient API.
  Optional<ArrayRef<int64_t>> getOptionalSizes() const;

  /// Get the raw nullable Type representing the dtype of this tensor type.
  ///
  /// It is expected that for many users, `hasDtype`/`getDtype` will be a more
  /// convenient API.
  Type getOptionalDtype() const;

  /// Return true if this type has a list of sizes.
  bool hasSizes() const { return getOptionalSizes().has_value(); }

  /// Get the list of sizes. Requires `hasSizes()`.
  ArrayRef<int64_t> getSizes() const {
    assert(hasSizes() && "must have sizes");
    return getOptionalSizes().value();
  }

  /// Return true if all sizes of this tensor are known.
  bool areAllSizesKnown() const {
    return hasSizes() && llvm::all_of(getSizes(), [](int64_t size) {
             return size != kUnknownSize;
           });
  }

  /// Return true if this type has a known dtype.
  bool hasDtype() const { return static_cast<bool>(getOptionalDtype()); }

  /// Get the dtype. Requires `hasDtype()`.
  Type getDtype() const {
    assert(hasDtype() && "must have a dtype");
    return getOptionalDtype();
  }

  /// Enable isa/dyn_cast for BaseTensorType.
  static bool classof(Type type);
};

using GetTensorTypeFn =
    llvm::function_ref<Type(MLIRContext *, Optional<ArrayRef<int64_t>>, Type)>;

} // namespace Paddle
} // namespace paddle
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Paddle/PaddleTypes.h.inc"

// inline methods
namespace mlir {
namespace paddle {
namespace Paddle {

inline Optional<ArrayRef<int64_t>> BaseTensorType::getOptionalSizes() const {
  if (auto tensor = dyn_cast<NonValueTensorType>())
    return tensor.getOptionalSizes();
  llvm_unreachable("not a BaseTensorType!");
}

inline Type BaseTensorType::getOptionalDtype() const {
  if (auto tensor = dyn_cast<NonValueTensorType>())
    return tensor.getOptionalDtype();
  llvm_unreachable("not a BaseTensorType!");
}

inline bool BaseTensorType::classof(Type type) {
  return type.isa<NonValueTensorType>();
}

} // namespace Paddle
} // namespace paddle
} // namespace mlir

#endif // F109FBEB_EEAF_4D85_8D01_AC401C3AD5F5
