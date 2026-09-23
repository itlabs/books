//===- pix-opt.cpp - pix 方言的 opt 驱动 ------------------------*- C++ -*-===//
//
// 这就是"你自己的 mlir-opt"：同一套命令行、同一套 pass 管线语法，只是多认识 pix。
// 全部内容只有两步：注册东西，然后把命令行交给 MlirOptMain。
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// 第 12 章：一条命名的 pass 管线
//
// 第 11 章之后，"把 pix 化简干净"这件事要敲三个开关：--canonicalize --cse
// --symbol-dce。把它包成一个名字，读者和测试都少写一串，而且这个名字是
// 我们方言的**公开接口**：以后管线内容变了，用它的人不用改命令。
//
// 注意它是"管线"而不是"pass"——没有新写任何 Pass 类，只是把现成的串起来。
// 第 13 章写真正的 pix 自己的 Pass 时，就加到这个函数里。
//===----------------------------------------------------------------------===//
static void buildPixSimplifyPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  // 1. 注册 pass：内建的那些（--canonicalize、--cse……）拿来就用。
  mlir::registerAllPasses();
  // pix 自己的 pass（第 13 章）。这一个函数由 -gen-pass-decls -name Pix 生成，
  // Passes.td 里加了新 pass 就自动包含进来，不用在这里逐个登记。
  mlir::pix::registerPixPasses();

  // 我们自己那条管线，注册成 --pix-simplify。
  // 静态对象的构造函数完成注册，所以放在 main 里也行、放全局也行；
  // 上游习惯放全局，这里放 main 是为了让"注册"这件事在阅读顺序上更明显。
  static mlir::PassPipelineRegistration<> pixSimplify(
      "pix-simplify", "把 pix IR 化简干净：canonicalize + cse + symbol-dce",
      buildPixSimplifyPipeline);

  // 2. 注册方言：registry 决定了这个工具"能解析哪些 op"。
  //    只注册需要被**解析**的方言即可；pix 必须有，其余内建方言一并带上，
  //    因为 pix 会和 arith/tensor/func 混在同一个模块里（第 6 章讲的方言共存）。
  mlir::DialectRegistry registry;
  registry.insert<mlir::pix::PixDialect>();
  mlir::registerAllDialects(registry);

  // 第 18 章补的一行。transform 方言的**大部分 op 住在扩展里**，不在方言本体里：
  // transform.structured.match / tile_using_for 由 linalg 的 transform 扩展提供。
  // 只注册方言不注册扩展，写那些 op 会得到
  //   error: custom op 'transform.structured.match' is unknown
  // 这个报错很容易被误读成"版本里没这个 op"。
  mlir::registerAllExtensions(registry);

  // 第 25 章：把 MLIR 方言翻译成 LLVM IR 需要另一套接口注册
  // （LLVMTranslationDialectInterface）。--gpu-module-to-binary 这类
  // "要真的生成机器码"的 pass 依赖它；少了它报的是
  //   missing `LLVMTranslationDialectInterface` registration for dialect
  //   for op: gpu.module
  // 注意它和 registerAllDialects / registerAllExtensions 是三件不同的事。
  mlir::registerAllToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "pix optimizer driver\n", registry));
}
