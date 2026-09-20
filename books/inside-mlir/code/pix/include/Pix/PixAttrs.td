//===- PixAttrs.td - pix 方言的自定义属性 ------------------*- tablegen -*-===//
//
// 《深入 MLIR》第 9 章：一个枚举属性 #pix.border<clamp|zero|wrap>，
// 描述卷积到了图像边界之外该怎么取值。
//
//===----------------------------------------------------------------------===//

#ifndef PIX_ATTRS
#define PIX_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "Pix/PixDialect.td"

// 第一步：定义枚举本身（C++ 侧会生成一个 enum class BorderMode）。
def Pix_BorderMode : I32EnumAttr<"BorderMode", "卷积的边界处理方式", [
      I32EnumAttrCase<"Clamp", 0, "clamp">,  // 取最近的边界像素
      I32EnumAttrCase<"Zero",  1, "zero">,   // 界外当 0
      I32EnumAttrCase<"Wrap",  2, "wrap">    // 环绕到另一侧
    ]> {
  let cppNamespace = "::mlir::pix";
  // 关掉"顺带生成一个属性类"，因为下一步我们要用 EnumAttr 生成更好用的那种
  // （带方言前缀、能自己解析打印的 #pix.border<clamp>）。
  let genSpecializedAttr = 0;
}

// 第二步：把枚举包装成一个真正的方言属性。
// 这样它就有了自己的 mnemonic（border）与打印格式 #pix.border<clamp>。
def Pix_BorderModeAttr : EnumAttr<Pix_Dialect, Pix_BorderMode, "border"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// 第 12 章：第二个枚举属性 —— pix.reduce 的归约方式
//   第 8 章把 reduce 的语义写死成"求和"，留了句"以后用枚举扩展"。现在兑现。
//   写法和 border 一模一样（两步），这正是要点：枚举属性是有套路的，
//   第二个就该是照抄。
//===----------------------------------------------------------------------===//

def Pix_ReduceKind : I32EnumAttr<"ReduceKind", "全图归约的方式", [
      I32EnumAttrCase<"Sum", 0, "sum">,
      I32EnumAttrCase<"Max", 1, "max">,
      I32EnumAttrCase<"Min", 2, "min">
    ]> {
  let cppNamespace = "::mlir::pix";
  let genSpecializedAttr = 0;
}

def Pix_ReduceKindAttr : EnumAttr<Pix_Dialect, Pix_ReduceKind, "reduce_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // PIX_ATTRS
