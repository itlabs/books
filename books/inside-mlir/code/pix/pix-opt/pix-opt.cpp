//===- pix-opt.cpp - pix 方言的 opt 驱动 ------------------------*- C++ -*-===//
//
// 这就是"你自己的 mlir-opt"：同一套命令行、同一套 pass 管线语法，只是多认识 pix。
// 全部内容只有两步：注册东西，然后把命令行交给 MlirOptMain。
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Pix/PixDialect.h"

int main(int argc, char **argv) {
  // 1. 注册 pass：内建的那些（--canonicalize、--cse……）拿来就用。
  //    以后 pix 自己的 pass（第 13 章起）也在这里注册。
  mlir::registerAllPasses();

  // 2. 注册方言：registry 决定了这个工具"能解析哪些 op"。
  //    只注册需要被**解析**的方言即可；pix 必须有，其余内建方言一并带上，
  //    因为 pix 会和 arith/tensor/func 混在同一个模块里（第 6 章讲的方言共存）。
  mlir::DialectRegistry registry;
  registry.insert<mlir::pix::PixDialect>();
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "pix optimizer driver\n", registry));
}
