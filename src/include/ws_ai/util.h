#pragma once
#include <string>

namespace ws_ai {

std::string uuid4();

bool ensure_dir(const std::string& path);

// 将二进制写入文件（覆盖）
bool write_file_binary(const std::string& path, const std::string& bytes);

// 读文件（用于静态文件服务）
bool read_file_binary(const std::string& path, std::string& out);

// 简单的 content-type 推断
std::string guess_mime(const std::string& path);

// 路径拼接（非常简化，不处理复杂边界）
std::string join_path(const std::string& a, const std::string& b);

} // namespace ws_ai