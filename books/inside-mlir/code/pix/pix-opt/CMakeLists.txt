set(LIBS
  MLIRPix                  # 我们自己的方言
  MLIRPixTransforms        # 第 13 章：我们自己的 pass
  MLIROptLib               # mlir-opt 的主体逻辑（命令行、管线、文件读写）
  MLIRRegisterAllDialects   # 注册全部内建方言，好让 pix 能和它们混用
  MLIRRegisterAllPasses     # 注册全部内建 pass，--canonicalize 等直接可用
  MLIRRegisterAllExtensions # 第 18 章：transform 方言的 op 住在扩展里
  )

add_llvm_executable(pix-opt pix-opt.cpp)
llvm_update_compile_flags(pix-opt)
target_link_libraries(pix-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(pix-opt)
