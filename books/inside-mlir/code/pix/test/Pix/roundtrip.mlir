// RUN: pix-opt %s | pix-opt | FileCheck %s
//
// 一条 RUN 行验两件事：pix-opt 能解析 pix.add，且它打印出来的东西还能再被解析回去
// （round-trip，第 3 章）。管线里第二个 pix-opt 就是为了这个。

// CHECK-LABEL: func.func @add_two_images
func.func @add_two_images(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: pix.add %{{.*}}, %{{.*}} : tensor<4x4xf32>
  %sum = pix.add %a, %b : tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @dynamic_shape
func.func @dynamic_shape(%a: tensor<?x4xf32>, %b: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: pix.add %{{.*}}, %{{.*}} : tensor<?x4xf32>
  %sum = pix.add %a, %b : tensor<?x4xf32>
  return %sum : tensor<?x4xf32>
}
