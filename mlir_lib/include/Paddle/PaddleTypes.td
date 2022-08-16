#ifndef A267F280_AF72_4448_BC0A_C2AE6F5E16F0
#define A267F280_AF72_4448_BC0A_C2AE6F5E16F0

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "PaddleDialect.td"

class Paddle_Type<string name, string typeMnemonic, string baseCppClass = "::mlir::Type">
    : TypeDef<Paddle_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Paddle_IntType : Paddle_Type<"Int", "int"> {
  let summary = "Paddle IntType";
  let description = [{
    Integer type.
  }];
}

// For standard ArrayRefs, which require allocation.
class OptionalArrayRefParameter<string arrayOf, string desc = ""> :
    AttrOrTypeParameter<
      "::llvm::Optional<::llvm::ArrayRef<" # arrayOf # ">>", desc> {
  let allocator = [{
    if ($_self.has_value()) {
      $_dst.value() = $_allocator.copyInto($_self.value());
    }
  }];
}

class AnyPaddleTensorType<string name, string typeMnemonic>
    : Paddle_Type<name, typeMnemonic, "::mlir::paddle::Paddle::BaseTensorType"> {
  let summary = "Multi-dimensional array modeling Paddle's Tensor type";
  let description = [{
    Syntax:
    ```
    tensor-type ::= (`!paddle.tensor` | `!paddle.vtensor`) tensor-modifiers?
    tensor-modifiers ::= `<` sizes-spec `,` dtype-spec `>`
    sizes-spec ::= `*` | `[` size-list `]`
    size-list ::= /*empty*/ | size-list-nonempty
    size-list-nonempty = size (`,` size)*
    size ::= `?` | decimal-literal
    dtype-spec ::= `unk` | type
    ```
  }];
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t", "sizes of dimensions">:$optionalSizes,
    "::mlir::Type":$optionalDtype
  );
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  string extraBaseClassDeclaration = [{
  }];
}

def Paddle_NonValueTensorType : AnyPaddleTensorType<"NonValueTensor", "tensor"> {
}

def AnyPaddleTensorType : Type<
    CPred<"$_self.isa<::mlir::paddle::Paddle::BaseTensorType>()">,
    "Any Paddle tensor type"
>;

#endif // A267F280_AF72_4448_BC0A_C2AE6F5E16F0
