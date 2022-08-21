#ifndef A267F280_AF72_4448_BC0A_C2AE6F5E16F0
#define A267F280_AF72_4448_BC0A_C2AE6F5E16F0

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
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

// TODO(allen) replace with !paddle.tensor
// !paddle.tensor should have same i/o format as built-in tensor
// def Paddle_Tensor : TypeAlias<AnyRankedTensor>;
def Paddle_Tensor : TypeAlias<AnyTensor>;

#endif // A267F280_AF72_4448_BC0A_C2AE6F5E16F0
