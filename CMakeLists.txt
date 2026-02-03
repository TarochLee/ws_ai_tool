cmake_minimum_required(VERSION 3.20)

# 关键：增加 OBJC，让 .m 用 Objective-C 编译
project(ws_ai_tool LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)

# 强制 ggml-metal 的 .m 按 OBJC 编译，避免被当成 OBJCXX
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m
    PROPERTIES
    LANGUAGE OBJC
)
add_subdirectory(src)