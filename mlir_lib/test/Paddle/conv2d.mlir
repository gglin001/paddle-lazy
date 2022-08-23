// RUN: paddle-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func.func @bar(
    func.func @bar(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>){
        %1 = paddle.conv2d %arg0 %arg1 {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
        return
    }
}
