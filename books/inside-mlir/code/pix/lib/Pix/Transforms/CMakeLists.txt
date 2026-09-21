# 第 15 章：DRR。-gen-rewriters 把 Pat<> 编成 C++ pattern 类，
# 生成的 .inc 由 DRRPass.cpp include 进去。
set(LLVM_TARGET_DEFINITIONS PixPatterns.td)
mlir_tablegen(PixPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRPixPatternsIncGen)

# 生成的 PixPatterns.inc 落在**本目录对应的构建目录**里，不在 include/ 下，
# 所以要把它加进搜索路径，否则 DRRPass.cpp 里那句 #include 找不到它。
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# 第 13 章：pass 单独成一个库。和方言库分开是上游的惯例——
# 只想用方言（比如做个解析器）的人不必链进所有变换代码。
add_mlir_dialect_library(MLIRPixTransforms
  CostPasses.cpp
  RewriteAdd.cpp
  DRRPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PIX_SOURCE_DIR}/include/Pix/Transforms

  DEPENDS
  MLIRPixTransformsPassIncGen
  MLIRPixPatternsIncGen      # 第 15 章的 DRR

  LINK_LIBS PUBLIC
  MLIRPix                   # 我们自己的方言
  MLIRFuncDialect           # pass 作用在 func.func 上
  MLIRIR
  MLIRPass
  MLIRArithDialect           # 第 14 章的 pattern 要造 arith.constant
  MLIRTransformUtils         # applyPatternsGreedily
  )
