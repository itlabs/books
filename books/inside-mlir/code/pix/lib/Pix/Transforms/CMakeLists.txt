# 第 13 章：pass 单独成一个库。和方言库分开是上游的惯例——
# 只想用方言（比如做个解析器）的人不必链进所有变换代码。
add_mlir_dialect_library(MLIRPixTransforms
  CostPasses.cpp
  RewriteAdd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PIX_SOURCE_DIR}/include/Pix/Transforms

  DEPENDS
  MLIRPixTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPix                   # 我们自己的方言
  MLIRFuncDialect           # pass 作用在 func.func 上
  MLIRIR
  MLIRPass
  MLIRArithDialect           # 第 14 章的 pattern 要造 arith.constant
  MLIRTransformUtils         # applyPatternsGreedily
  )
