// 第 21 章：tiling、融合、halo。三条 RUN 行各驱动一段 transform 调度。
//
// RUN: pix-opt %s --pix-to-linalg --transform-interpreter --canonicalize --cse | FileCheck %s
//
// 这个文件里有三个 payload 函数和三段调度，靠 transform.target_tag 之类不好分，
// 所以用一段 schedule 按函数名匹配。

module attributes {transform.with_named_sequence} {

  //=== 融合：两步逐点运算合成一个循环嵌套，中间那张整图消失 ===//
  //
  // CHECK-LABEL: func.func @fused
  // 只有一对循环，不是两对
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // 两个 generic 都在里面，后者直接吃前者的结果（不经过整张中间图）
  // CHECK: %[[P:.*]] = linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[P]]
  // CHECK: arith.mulf
  func.func @fused(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %f: f32) -> tensor<8x8xf32> {
    %s = pix.add %a, %b : tensor<8x8xf32>
    %m = pix.scale %s, %f : tensor<8x8xf32>
    return %m : tensor<8x8xf32>
  }

  //=== halo：stencil 的输入切片比输出切片大 ===//
  //
  // CHECK-LABEL: func.func @halo
  // 先 pad 一圈半径（same 语义）
  // CHECK: tensor.pad
  // CHECK-SAME: low[1, 1] high[1, 1]
  // 输入切片 6x6，输出切片 4x4 —— 那两圈就是 halo，由 indexing map 自动算出
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [6, 6] [1, 1]
  // CHECK-SAME: to tensor<6x6xf32>
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [4, 4] [1, 1]
  // CHECK-SAME: to tensor<4x4xf32>
  func.func @halo(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %k = pix.kernel dense<1.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
    %c = pix.convolve %img, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
    return %c : tensor<8x8xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // ① 融合：拿 @fused 里的两个 generic，tile 消费者并把生产者融进去
    %f = transform.structured.match ops{["func.func"]}
         attributes{sym_name = "fused"} in %root
      : (!transform.any_op) -> !transform.any_op
    %gs = transform.structured.match ops{["linalg.generic"]} in %f
      : (!transform.any_op) -> !transform.any_op
    %prod, %cons = transform.split_handle %gs
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %l1:2 = transform.structured.fuse %cons tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ② halo：只 tile 前两维（输出行列），归约维写 0 表示不切
    %h = transform.structured.match ops{["func.func"]}
         attributes{sym_name = "halo"} in %root
      : (!transform.any_op) -> !transform.any_op
    %hg = transform.structured.match ops{["linalg.generic"]} in %h
      : (!transform.any_op) -> !transform.any_op
    %htiled, %l2:2 = transform.structured.tile_using_for %hg tile_sizes [4, 4, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
