// RUN: pix-opt %s --transform-interpreter | FileCheck %s
//
// 第 18 章：把"什么时候做什么变换"写成 IR。
// 这个文件里 payload（func.func）和 schedule（transform.named_sequence）
// 住在同一个 module 里——上游测试也是这么写的，省掉一个 --transform-preload-library。
//
// 注意 pix-opt 必须注册 **extensions**（registerAllExtensions），
// 因为 transform.structured.* 这些 op 住在 linalg 的 transform 扩展里，
// 不在 transform 方言本体里。少了那一句会得到
//   error: custom op 'transform.structured.match' is unknown

module attributes {transform.with_named_sequence} {

  // 被变换的对象（payload）
  // CHECK-LABEL: func.func @driven_by_schedule
  // CHECK-SAME: pix.elements = 16 : i64
  // CHECK-SAME: pix.num_ops = 1 : i64
  func.func @driven_by_schedule(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %r = pix.add %a, %b : tensor<4x4xf32>
    return %r : tensor<4x4xf32>
  }

  // 调度本身（schedule）。它是 IR，不是命令行参数——这就是 transform 方言的全部要点。
  // CHECK: transform.named_sequence @__transform_main
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // ① 匹配：拿到一个 handle，指向所有 func.func
    %fns = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // ② 对它们跑我们自己的 pass（第 13 章写的那个）。
    //    apply_registered_pass 让 transform 脚本能驱动任何已注册的 pass，
    //    于是"调度"和"pass 实现"彻底解耦。
    %after = transform.apply_registered_pass "pix-cost-report" to %fns
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
