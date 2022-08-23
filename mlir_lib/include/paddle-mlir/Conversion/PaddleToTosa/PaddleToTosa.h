#ifndef B72F1D29_D540_4196_83F1_32607DDD8A81
#define B72F1D29_D540_4196_83F1_32607DDD8A81

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace paddle {

std::unique_ptr<Pass> createPaddleToTosaPass();

} // namespace paddle
} // namespace mlir

#endif // B72F1D29_D540_4196_83F1_32607DDD8A81
