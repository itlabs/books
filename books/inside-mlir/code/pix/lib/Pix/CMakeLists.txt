# add_mlir_dialect_library 比 add_library 多做几件事：登记到 MLIR 的库清单里
# （于是 mlir_check_all_link_libraries 能检查链接完整性）、处理 DEPENDS 的
# TableGen 目标、以及统一编译选项。
add_mlir_dialect_library(MLIRPix
  PixAttrs.cpp
  PixDialect.cpp
  PixOps.cpp
  PixTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PIX_SOURCE_DIR}/include/Pix

  DEPENDS
  MLIRPixOpsIncGen          # ← add_mlir_dialect(PixOps pix) 建立的目标
  MLIRPixAttrsIncGen        # ← 第 9 章手写的那组 mlir_tablegen 建立的目标

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect          # 第 11 章的 canonicalize pattern 要造 arith.constant/mulf
  MLIRInferTypeOpInterface  # SameOperandsAndResultType 用到
  MLIRSideEffectInterfaces  # Pure 用到
  )
