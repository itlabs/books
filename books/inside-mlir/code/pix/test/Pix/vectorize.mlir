// 第 23 章：向量化。两条 RUN 行对照"能整除"与"不能整除"两种情形。
//
// RUN: pix-opt %s --pix-to-linalg --transform-interpreter --canonicalize | FileCheck %s

module attributes {transform.with_named_sequence} {

  // tile 4x4 能整除 8x8 → 每个 tile 都是满的，in_bounds 全 true，不需要 mask
  // CHECK-LABEL: func.func @vec
  // CHECK: vector.transfer_read
  // CHECK-SAME: in_bounds = [true, true]
  // CHECK-SAME: vector<4x4xf32>
  // 标量 arith.addf 变成了**向量上的** arith.addf —— 同一个 op，换了类型
  // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<4x4xf32>
  // CHECK: vector.transfer_write
  // 一个 mask 都没有
  // CHECK-NOT: vector.create_mask
  func.func @vec(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %s = pix.add %a, %b : tensor<8x8xf32>
    return %s : tensor<8x8xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %l:2 = transform.structured.tile_using_for %g tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // vector_sizes 要和 tile 大小对上 —— 向量化是在 tile 内部做的
    transform.structured.vectorize %tiled vector_sizes [4, 4] : !transform.any_op
    transform.yield
  }
}
