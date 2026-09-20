# lit 测试（第 12 章细讲）。这里做三件事：
#   1. 把 lit.site.cfg.py.in 里的 @占位符@ 替换成真实路径，生成 lit.site.cfg.py
#   2. 声明测试依赖：要先编出 pix-opt、FileCheck 等工具才能跑测试
#   3. 建立 check-pix 目标：ninja check-pix 就跑全套
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PIX_TEST_DEPENDS
  FileCheck count not
  pix-opt
  )

add_lit_testsuite(check-pix "Running the pix regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PIX_TEST_DEPENDS}
  )
set_target_properties(check-pix PROPERTIES FOLDER "Tests")

add_lit_testsuites(PIX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PIX_TEST_DEPENDS})
