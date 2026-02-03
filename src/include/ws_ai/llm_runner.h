#pragma once
#include <functional>
#include <string>

namespace ws_ai {

struct GenParams {
    int max_new_tokens = 800;     // 输出太短就加大
    int min_new_tokens = 250;     // 不到这个长度不允许 EOS 结束（配合重采样）
    int max_resample_eos = 128;   // 早 EOS 的最大重采样次数

    float top_p = 0.9f;
    int   top_k = 40;
    float temp  = 0.6f;
};

struct LLMResult {
    bool ok = false;
    std::string text;
    std::string error;
};

// on_delta: 每产生一段文本就回调（用于流式累积 + 进度推进）
// 返回 ok/error
LLMResult run_llm_summarize(const std::string& model_path,
                            const std::string& prompt,
                            const GenParams& params,
                            const std::function<void(const std::string&)>& on_delta);

} // namespace ws_ai