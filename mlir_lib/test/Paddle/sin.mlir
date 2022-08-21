// RUN: paddle-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func.func @bar(
    func.func @bar(%arg0: tensor<1x3x2x2xf32>) {
        %1 = paddle.sin %arg0 : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
        return
    }
    // CHECK-LABEL: func.func @baz(
    func.func @baz() {
        %0 = arith.constant dense<1.000000e+00> : tensor<2xf32>
        %1 = paddle.sin %0 : (tensor<2xf32>) -> tensor<2xf32>
        return
    }
    // CHECK-LABEL: func.func @lala(
    func.func @lala() {
        %0 = paddle.constant(dense<1.0> : tensor<f32>) : tensor<*xf32>
        %1 = paddle.sin %0 : (tensor<*xf32>) -> tensor<*xf32>
        return
    }
}
