#ifndef EC342222_0958_42BB_99C7_461FADCB026D
#define EC342222_0958_42BB_99C7_461FADCB026D

include "PaddleDialect.td"
include "PaddleTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Paddle_AddOp : Paddle_Op<"add", [NoSideEffect]> {
    let summary = "Illustrates how to define an operation.";
    let arguments = (ins Paddle_IntType:$x, Paddle_IntType:$y);
    let results = (outs Paddle_IntType:$res);
    let assemblyFormat = [{
        $x $y attr-dict `:` functional-type(operands, results)
    }];

    // let hasFolder = 1;
    // let hasCanonicalizer = 1;
    // let hasCustomAssemblyFormat = 1;
}

def Paddle_ConstantOp : Paddle_Op<"constant", [ConstantLike, NoSideEffect]> {
    let summary = "Constant op.";
    let arguments = (ins ElementsAttr:$value);
    let results = (outs Paddle_Tensor:$result);
    let assemblyFormat = [{
        `(` $value `)` attr-dict `:` qualified(type($result))
    }];
}

def Paddle_ReluOp : Paddle_Op<"relu", [NoSideEffect]> {
    let summary = "Relu op.";
    let arguments = (ins Paddle_Tensor:$x);
    let results = (outs Paddle_Tensor:$res);
    let assemblyFormat = [{
        $x attr-dict `:` functional-type(operands, results)
    }];
}

def Paddle_SinOp : Paddle_Op<"sin", [NoSideEffect]> {
    let summary = "Sin op.";
    let arguments = (ins Paddle_Tensor:$x);
    let results = (outs Paddle_Tensor:$res);
    let assemblyFormat = [{
        $x attr-dict `:` functional-type(operands, results)
    }];
}

#endif // EC342222_0958_42BB_99C7_461FADCB026D
