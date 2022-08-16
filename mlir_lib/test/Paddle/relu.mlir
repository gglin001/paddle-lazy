// RUN: paddle-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func.func @bar(
    func.func @bar(%arg0: !paddle.tensor<*,f32>) -> !paddle.tensor {
        %1 = paddle.relu %arg0 : (!paddle.tensor<*,f32>) -> !paddle.tensor
        return %1 : !paddle.tensor
    }
}
