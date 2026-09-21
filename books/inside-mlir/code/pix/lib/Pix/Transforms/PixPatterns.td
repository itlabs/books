//===- PixPatterns.td - 第 15 章：用 DRR 声明式地写 pattern -*- tablegen -*-===//
//
// DRR（Declarative Rewrite Rule）把第 14 章那种 C++ pattern 压成一行 `Pat<>`。
// 这个文件刻意排成"从最舒服到最别扭"的顺序，好看清它的天花板在哪：
//   ① 纯结构改写           —— DRR 的主场，一行顶三十行 C++
//   ② 带约束的结构改写      —— 加一个 Constraint，仍然很舒服
//   ③ 嵌套结果 + 建新 op    —— 还行，但已经要理解 DAG 怎么写
//   ④ 需要造常量           —— 得靠 NativeCodeCall 逃回 C++，收益开始变负
//
// 放在 lib/ 而不是 include/：这些 pattern 是实现细节，没人需要 include 它们。
// 上游也是这么放的（mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td）。
//
//===----------------------------------------------------------------------===//

#ifndef PIX_PATTERNS
#define PIX_PATTERNS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "Pix/PixOps.td"

//===----------------------------------------------------------------------===//
// ① 纯结构改写：转置两次等于没转
//===----------------------------------------------------------------------===//

// 读法：左边是要匹配的 DAG（外层 transpose 套内层 transpose），
// 右边 replaceWithValue 表示"用这个已有的值替换整个匹配"。
// 这一行等价于第 11 章那个十几行的 TransposeOp::fold。
def TransposeTwiceIsNop : Pat<
  (Pix_TransposeOp (Pix_TransposeOp $x)),
  (replaceWithValue $x)>;

//===----------------------------------------------------------------------===//
// ② 带约束：只有形状真的绕回来了才消
//
// 上面那条其实不够严谨——两次转置的结果类型必须和原输入一致才能替换。
// 第 11 章的 fold 里我们手写了一个 if；DRR 里写成一个 Constraint。
//===----------------------------------------------------------------------===//

def SameTypeAs : Constraint<
  CPred<"$0.getType() == $1.getType()">, "两个值类型相同">;

def TransposeTwiceChecked : Pat<
  (Pix_TransposeOp:$outer (Pix_TransposeOp $x)),
  (replaceWithValue $x),
  [(SameTypeAs $x, $outer)]>;

//===----------------------------------------------------------------------===//
// ③ 第一道摩擦：目标 op 带着你手里没有的属性
//
// 想写成纯声明式的 (Arith_MulFOp $a, $b)，mlir-tblgen 会拒绝：
//   error: op 'arith.mulf' argument number mismatch: 2 in pattern vs. 4 in definition
// 因为 arith.mulf 的 arguments 是四个：lhs、rhs、fastmath、roundingmode。
// DRR 建目标 op 时**必须按 ODS 里的 arguments 给全**，即使后两个有默认值。
//
// 于是只能用 NativeCodeCall 逃回 C++ 一行——那一行正好用上 op 自带的
// 三参数 builder，把 fastmath/roundingmode 交给默认值。
// 这是 DRR 第一次"不划算"：声明式的可读性开始被样板抵消。
//===----------------------------------------------------------------------===//

def MakeMulF : NativeCodeCall<
  "::mlir::arith::MulFOp::create($_builder, $_builder.getUnknownLoc(), $0, $1)">;

def MergeScalesDRR : Pat<
  (Pix_ScaleOp (Pix_ScaleOp $x, $a), $b),
  (Pix_ScaleOp $x, (MakeMulF $a, $b))>;

//===----------------------------------------------------------------------===//
// ④ 天花板：需要一个**凭空造出来的常量**
//
// add(%x, %x) → scale(%x, 2.0)。那个 2.0 不在输入里，DRR 的 DAG 语法
// 表达不出"造一个 f32 常量 2.0"，只能用 NativeCodeCall 写一段 C++ 表达式。
// 到这一步，声明式的可读性优势基本抵消了——第 15 章正文会讨论这条界线。
//===----------------------------------------------------------------------===//

def MakeF32Two : NativeCodeCall<
  "$_builder.getF32FloatAttr(2.0)">;

def SameValue : Constraint<CPred<"$0 == $1">, "两个值是同一个">;

def AddSelfToScaleDRR : Pat<
  (Pix_AddOp:$op $x, $y),
  (Pix_ScaleOp $x, (Arith_ConstantOp (MakeF32Two))),
  [(SameValue $x, $y)]>;

#endif // PIX_PATTERNS
