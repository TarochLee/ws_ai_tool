cmake_minimum_required(VERSION 3.20)

add_executable(ws_ai_server
    src/main.cpp
    src/http_server.cpp
    src/job_manager.cpp
    src/llm_runner.cpp
    src/prompt.cpp
    src/util.cpp
    src/ocr_vision.mm
    src/pipeline.mm
)

target_include_directories(ws_ai_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src/../include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src/../include
)

target_link_libraries(ws_ai_server PRIVATE
    llama
)

find_library(FW_FOUNDATION Foundation)
find_library(FW_VISION Vision)
find_library(FW_COREGRAPHICS CoreGraphics)
find_library(FW_IMAGEIO ImageIO)

target_link_libraries(ws_ai_server PRIVATE
    ${FW_FOUNDATION}
    ${FW_VISION}
    ${FW_COREGRAPHICS}
    ${FW_IMAGEIO}
)

# 可选：确保 .mm 用 OBJCXX
set_source_files_properties(src/ocr_vision.mm src/pipeline.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
)