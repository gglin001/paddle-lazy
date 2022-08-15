#ifndef A267F280_AF72_4448_BC0A_C2AE6F5E16F0
#define A267F280_AF72_4448_BC0A_C2AE6F5E16F0

include "mlir/IR/AttrTypeBase.td"
include "PaddleDialect.td"

class Paddle_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Paddle_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Paddle_IntType : Paddle_Type<"Int", "int"> {
  let summary = "Paddle IntType";
  let description = [{
    Integer type.
  }];
}

#endif // A267F280_AF72_4448_BC0A_C2AE6F5E16F0
