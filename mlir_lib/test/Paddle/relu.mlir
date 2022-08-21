// RUN: paddle-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func.func @bar(
    func.func @bar(%arg0: tensor<?xf32>) -> tensor<?xf32> {
        %1 = paddle.relu %arg0 : (tensor<?xf32>) -> tensor<?xf32>
        return %1 : tensor<?xf32>
    }

    // CHECK-LABEL: func.func @baz(
    func.func @baz() {
        %0 = paddle.constant(dense<1.0> : tensor<f32>) : tensor<f32>
        %1 = paddle.relu %0 : (tensor<f32>) -> tensor<f32>
        return
    }
}
