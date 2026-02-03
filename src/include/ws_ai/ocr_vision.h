#pragma once
#include <string>

namespace ws_ai {

// 使用 Vision OCR：输入图片路径 -> 输出识别文本（UTF-8）
// 失败返回空字符串
std::string ocr_with_vision(const std::string& image_path);

} // namespace ws_ai