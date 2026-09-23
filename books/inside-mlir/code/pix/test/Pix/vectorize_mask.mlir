// 第 23 章：tile 大小**不能整除**时，向量化必须生成 mask。
// 这是 vector 方言存在 mask 的根本原因：向量宽度是固定的，数据边界不是。
//
// RUN: pix-opt %s --pix-to-linalg --transform-interpreter --canonicalize | FileCheck %s

module attributes {transform.with_named_sequence} {

  // tile 3x3 除不尽 8x8 → 最后一块是残缺的
  // CHECK-LABEL: func.func @masked
  // 切片类型变成动态的，因为最后一块尺寸不同
  // CHECK: tensor<?x?xf32>
  // mask 由运行时的实际尺寸算出来
  // CHECK: vector.create_mask %{{.*}}, %{{.*}} : vector<3x3xi1>
  // 读写都被包在 vector.mask 里（整行匹配，比 CHECK-SAME 稳）
  // CHECK: vector.mask %{{.*}} { vector.transfer_read
  // CHECK: vector.mask %{{.*}} { vector.transfer_write
  func.func @masked(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %s = pix.add %a, %b : tensor<8x8xf32>
    return %s : tensor<8x8xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %l:2 = transform.structured.tile_using_for %g tile_sizes [3, 3]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled vector_sizes [3, 3] : !transform.any_op
    transform.yield
  }
}
