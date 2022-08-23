#ifndef C8ACE83F_3034_40DD_988E_4AE8D75FD86B
#define C8ACE83F_3034_40DD_988E_4AE8D75FD86B

include "mlir/Pass/PassBase.td"

// PaddleToTosa
def PaddleToTosa : Pass<"paddle-to-tosa"> {
  let summary = "Lower PADDLE to the TOSA dialect";
  let description = [{
    Pass that converts Paddle operations to the equivalent TOSA operations.
  }];

  let constructor = "mlir::paddle::createPaddleToTosaPass()";
}

#endif // C8ACE83F_3034_40DD_988E_4AE8D75FD86B
