# add_mlir_dialect(<td 文件名不带后缀> <方言名>) 一句顶好几句：
# 它以 PixOps.td 为输入，生成
#   PixOps.h.inc / PixOps.cpp.inc             —— op 的声明与实现
#   PixOpsDialect.h.inc / PixOpsDialect.cpp.inc —— 方言类本身
# 并建立一个 TableGen 目标 MLIRPixOpsIncGen，供 lib/ 用 DEPENDS 挂上。
add_mlir_dialect(PixOps pix)

# 第 9 章：属性与枚举不在 add_mlir_dialect 的覆盖范围里（它只管 op / dialect / 类型），
# 所以这里手写底层的三步：设定输入 .td → 逐个生成器跑一遍 → 建一个 TableGen 目标。
# 这正是 add_mlir_dialect 内部做的事，只是换成我们自己控制。
set(LLVM_TARGET_DEFINITIONS PixAttrs.td)
mlir_tablegen(PixAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pix)
mlir_tablegen(PixAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pix)
mlir_tablegen(PixEnums.h.inc -gen-enum-decls)
mlir_tablegen(PixEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPixAttrsIncGen)

# 顺带生成方言文档（.md）。不是必需，但白拿的东西为什么不要。
add_mlir_doc(PixDialect PixDialect Pix/ -gen-dialect-doc)
add_mlir_doc(PixOps PixOps Pix/ -gen-op-doc)

# 第 13 章：pass 的 TableGen 在子目录里
add_subdirectory(Transforms)
