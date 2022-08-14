#ifndef E4168F65_A444_4D5E_9C51_763CEDDDD038
#define E4168F65_A444_4D5E_9C51_763CEDDDD038

include "mlir/IR/OpBase.td"

// Paddle dialect definition.
def Paddle_Dialect : Dialect {
    let name = "paddle";
    let summary = "A paddle out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::paddle";
    let useDefaultTypePrinterParser=1;
}

// Base paddle operation definition.
class Paddle_Op<string mnemonic, list<Trait> traits = []> :
        Op<Paddle_Dialect, mnemonic, traits>;

#endif // E4168F65_A444_4D5E_9C51_763CEDDDD038
