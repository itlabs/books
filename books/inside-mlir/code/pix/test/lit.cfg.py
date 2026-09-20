# -*- Python -*-
# pix 的 lit 配置（照上游 mlir/examples/standalone/test/lit.cfg.py 裁剪）
import os

import lit.formats
from lit.llvm import llvm_config

config.name = "PIX"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.pix_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.md"]

config.pix_tools_dir = os.path.join(config.pix_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# 让测试里写 pix-opt / mlir-opt 就能找到对应的可执行文件
tool_dirs = [config.pix_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["pix-opt", "mlir-opt"], tool_dirs)
