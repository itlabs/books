// 第 24 章：整条下降链走到底，然后**真的跑起来**。
//
// RUN: pix-opt %s \
// RUN:   --pix-to-linalg \
// RUN:   --one-shot-bufferize=bufferize-function-boundaries=true \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --expand-strided-metadata \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-cf-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:     --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
//
// (1+10)*2 = 22, (2+20)*2 = 44, (3+30)*2 = 66, (4+40)*2 = 88
// 这是全书第一次断言**运行结果**而不是 IR —— 前面所有章节验证的都是"IR 长什么样"，
// 这里验证的是"算出来的数对不对"。
//
// CHECK: Unranked Memref base@
// CHECK-SAME: rank = 2
// CHECK-SAME: offset = 0
// CHECK-SAME: sizes = [2, 2]
// CHECK-SAME: strides = [2, 1]
// CHECK: [22, 44]
// CHECK: [66, 88]

func.func @main() {
  %f = arith.constant 2.0 : f32
  %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = arith.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf32>
  %s = pix.add %a, %b : tensor<2x2xf32>
  %m = pix.scale %s, %f : tensor<2x2xf32>
  %u = tensor.cast %m : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(tensor<*xf32>)
