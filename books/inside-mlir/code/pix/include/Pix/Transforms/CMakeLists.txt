# 第 13 章：pass 的 TableGen。-gen-pass-decls 需要 -name 参数，
# 它决定生成的注册函数叫 registerPixPasses()。
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Pix)
add_public_tablegen_target(MLIRPixTransformsPassIncGen)

add_mlir_doc(Passes PixPasses Pix/ -gen-pass-doc)
