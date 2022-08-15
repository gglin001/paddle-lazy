// RUN: paddle-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func.func @bar(
    func.func @bar(%arg0: !paddle.int, %arg1: !paddle.int) {
        // CHECK: %{{.*}} = paddle.add %{{.*}} -> !paddle.int
        %res = paddle.add %arg0 %arg0 : (!paddle.int, !paddle.int) -> !paddle.int
        return
    }
}
