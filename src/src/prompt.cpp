#include "ws_ai/prompt.h"
#include <sstream>

namespace ws_ai {

std::string build_prompt(const std::string &ocr_text) {
    std::ostringstream oss;
    oss << "<|im_start|>system\n"
        << "你是一个严谨的技术助理。你只使用中文输出，且只输出两段：第一段为对截图文字的总结（不分点，一段话）；第二段为相关扩展知识（不分点，一段话）。不要输出额外标题、不要输出英文、不要输出提问或任务指令。\n"
        << "<|im_end|>\n"
        << "<|im_start|>user\n"
        << "以下是从截图OCR得到的文字，请基于文字内容完成总结与扩展。\n\n"
        << ocr_text << "\n"
        << "<|im_end|>\n"
        << "<|im_start|>assistant\n";
    return oss.str();
}

} // namespace ws_ai