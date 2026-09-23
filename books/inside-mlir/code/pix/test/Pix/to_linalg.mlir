// RUN: pix-opt %s --pix-to-linalg | FileCheck %s
//
// 第 20 章：pix → linalg。和第 16 章的 pix-to-arith 对照着看——
// 那条路更短，但把迭代空间丢了；这条路把它显式写出来，于是可 tile 可融合。

//===----------------------------------------------------------------------===//
// 逐点：indexing_maps 三个恒等映射，iterator_types 全 parallel
//===----------------------------------------------------------------------===//

// affine map 的别名全部打印在**文件最开头**，所以两个捕获都得放在
// 任何 CHECK-LABEL 之前 —— 放到函数中间的话游标已经过去了，会报
// "undefined variable"。
// CHECK-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[NONE:.*]] = affine_map<(d0, d1) -> ()>
//
// CHECK-LABEL: func.func @pointwise
// CHECK: tensor.empty()
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[ID]], #[[ID]], #[[ID]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK-NOT: pix.add
func.func @pointwise(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %s = pix.add %a, %b : tensor<4x4xf32>
  return %s : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// scale：标量**不进 ins**，直接在 region 里引用外面的值。
// linalg.generic 的 region 不是 IsolatedFromAbove，所以这么写合法，
// 而且省掉一次"把标量 splat 成整张图"的实体化。
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @scaled
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[ID]], #[[ID]]]
// ins 只有一个（图），标量 %arg1 是在 region 里直接用的
// CHECK-SAME: ins(%arg0 :
// CHECK: arith.mulf %in, %arg1
// CHECK-NOT: pix.scale
func.func @scaled(%a: tensor<4x4xf32>, %f: f32) -> tensor<4x4xf32> {
  %m = pix.scale %a, %f : tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 归约：**同一个 linalg.generic**，只是换了 map 和 iterator 类型。
// 输出的 map 是 (d0, d1) -> ()，两个维度全被归约掉。
// 这一条最能说明 indexing map 的表达力。
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reduction
// 累加器初值必须是归约的单位元 0
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[Z]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[ID]], #[[NONE]]]
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// %out 就是累加器当前值 —— destination-passing style 的体现
// CHECK: arith.addf %in, %out
// CHECK: tensor.extract
// CHECK-NOT: pix.reduce
func.func @reduction(%a: tensor<4x4xf32>) -> f32 {
  %r = pix.reduce %a : tensor<4x4xf32> -> f32
  return %r : f32
}
