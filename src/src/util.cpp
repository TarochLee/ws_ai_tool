#include "ws_ai/util.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace ws_ai {

std::string uuid4() {
    // 简化版 UUID：足够用于本地任务 id
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    uint64_t a = dis(gen);
    uint64_t b = dis(gen);

    std::ostringstream oss;
    oss << std::hex;
    oss << (a >> 32) << "-" << ((a >> 16) & 0xFFFF) << "-" << (a & 0xFFFF)
        << "-" << (b >> 48) << "-" << (b & 0xFFFFFFFFFFFFULL);
    return oss.str();
}

bool ensure_dir(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
}

bool write_file_binary(const std::string& path, const std::string& bytes) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(bytes.data(), (std::streamsize)bytes.size());
    return ofs.good();
}

bool read_file_binary(const std::string& path, std::string& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out = oss.str();
    return true;
}

std::string guess_mime(const std::string& path) {
    auto lower = path;
    for (auto& c : lower) c = (char)tolower(c);

    if (lower.size() >= 5 && lower.substr(lower.size()-5) == ".html")
        return "text/html; charset=utf-8";
    if (lower.size() >= 3 && lower.substr(lower.size()-3) == ".js")
        return "application/javascript; charset=utf-8";
    if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".css")
        return "text/css; charset=utf-8";
    if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".png")
        return "image/png";
    if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".jpg")
        return "image/jpeg";
    if (lower.size() >= 5 && lower.substr(lower.size()-5) == ".jpeg")
        return "image/jpeg";
    return "application/octet-stream";
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

} // namespace ws_ai