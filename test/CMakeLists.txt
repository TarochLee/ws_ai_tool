cmake_minimum_required(VERSION 3.20)
project(pic_brief LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

# 推荐：用新选项名（llama.cpp 提示 LLAMA_METAL deprecated）
set(GGML_METAL ON CACHE BOOL "Enable Metal backend" FORCE)

# 引入 llama.cpp submodule
add_subdirectory(llama.cpp)

# 你的可执行程序（Objective-C++，用于 Vision OCR + llama.cpp 推理）
add_executable(pic_brief main.mm)

# 强制写入 LC_RPATH，让 dyld 能找到 build/bin 下的 dylib
target_link_options(pic_brief PRIVATE
    "-Wl,-rpath,@executable_path/bin"
)

# 运行时动态库路径：让 pic_brief 在运行时去自身目录下的 bin/ 找 libllama/libggml*.dylib
set_target_properties(pic_brief PROPERTIES
    BUILD_RPATH "@executable_path/bin"
    INSTALL_RPATH "@executable_path/bin"
)

# macOS frameworks（OCR + CGImageRelease 等）
target_link_libraries(pic_brief
    PRIVATE
    "-framework Foundation"
    "-framework Vision"
    "-framework ImageIO"
    "-framework CoreGraphics"
    "-framework CoreFoundation"
)

# 链接 llama.cpp 生成的动态库目标
# 如果你 cmake 报 "Target llama not found"，执行：
#   cmake --build build --target help
# 然后把这里的 llama 替换成实际目标名。
target_link_libraries(pic_brief PRIVATE llama)