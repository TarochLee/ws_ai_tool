#include "ws_ai/llm_runner.h"

#include <algorithm>
#include <string>
#include <vector>

extern "C" {
#include "llama.h"
}

namespace ws_ai {

static inline void trim_inplace(std::string& s) {
    while (!s.empty() && (s.back()==' '||s.back()=='\n'||s.back()=='\r'||s.back()=='\t'))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t')) i++;
    s.erase(0, i);
}

static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    // 先探测长度
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                               nullptr, 0,
                               /*add_special*/ false,
                               /*parse_special*/ true);
    if (n < 0) n = -n;
    std::vector<llama_token> out((size_t)n);

    int32_t n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                out.data(), (int32_t)out.size(),
                                /*add_special*/ false,
                                /*parse_special*/ true);
    if (n2 < 0) out.clear();
    return out;
}

static std::string token_to_piece(const llama_vocab* vocab, llama_token tok) {
    std::string out;
    out.resize(256);

    // llama_token_to_piece(vocab, token, buf, length, lstrip, special)
    int32_t n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                                     /*lstrip*/ 0, /*special*/ true);
    if (n < 0) {
        out.resize((size_t)(-n));
        n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                                 /*lstrip*/ 0, /*special*/ true);
        if (n < 0) return {};
    }
    out.resize((size_t)n);
    return out;
}

// 不用 llama_batch_add，手动写 batch 结构
static void batch_add(llama_batch& b, llama_token tok, int32_t pos, int32_t seq, bool logits) {
    int32_t i = b.n_tokens;
    b.token[i] = tok;
    b.pos[i] = pos;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = seq;
    b.logits[i] = logits ? 1 : 0;
    b.n_tokens++;
}

LLMResult run_llm_summarize(const std::string& model_path,
                            const std::string& prompt,
                            const GenParams& params,
                            const std::function<void(const std::string&)>& on_delta) {
    LLMResult R;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        R.ok = false;
        R.error = "模型加载失败: " + model_path;
        llama_backend_free();
        return R;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 1024;
    // 注意：不要写 cp.flash_attn（你现在版本里已改名/不存在）
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        R.ok = false;
        R.error = "llama_context 创建失败";
        llama_model_free(model);
        llama_backend_free();
        return R;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto prompt_tokens = tokenize(vocab, prompt);
    if (prompt_tokens.empty()) {
        R.ok = false;
        R.error = "prompt tokenize 失败";
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return R;
    }

    // sampler chain（避免使用你版本里不存在的 repeat_penalty）
    llama_sampler* sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 1) 先 decode prompt，一次性喂入
    // 关键：只把“最后一个 token”的 logits 标记为 true，这样 n_outputs=1
    llama_batch b0 = llama_batch_init((int32_t)prompt_tokens.size(), 0, 1);
    b0.n_tokens = 0;
    for (int32_t i = 0; i < (int32_t)prompt_tokens.size(); ++i) {
        bool want_logits = (i == (int32_t)prompt_tokens.size() - 1);
        batch_add(b0, prompt_tokens[i], /*pos*/ i, /*seq*/ 0, want_logits);
    }

    if (llama_decode(ctx, b0) != 0) {
        llama_batch_free(b0);
        R.ok = false;
        R.error = "llama_decode(prompt) 失败";
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return R;
    }
    llama_batch_free(b0);

    // 重要修复：采样用 logits_id=0（因为本次 decode 只有 1 个 logits 输出）
    // 之前你用 token index 当 logits id，会触发 invalid logits id
    int32_t n_past = (int32_t)prompt_tokens.size();
    llama_token tok_eos = llama_vocab_eos(vocab);

    std::string out;
    out.reserve(8192);

    int eos_resample_used = 0;

    for (int step = 0; step < params.max_new_tokens; ++step) {
        // 本轮可采样的 logits 槽位：永远是 0（因为我们每轮 decode 也只请求一个 logits）
        llama_token tok = llama_sampler_sample(sampler, ctx, /*logits_id*/ 0);

        // 早停控制：太早 EOS -> 拒绝 EOS 重采样
        if (tok == tok_eos && step < params.min_new_tokens) {
            if (eos_resample_used < params.max_resample_eos) {
                eos_resample_used++;
                // 不 accept，直接再 sample 一次（注意：这会改变采样轨迹，但能显著减少“越来越短”）
                continue;
            }
            // 重采样次数耗尽：允许结束，避免死循环
        }

        // 正常接受 token
        llama_sampler_accept(sampler, tok);

        // 输出 piece
        std::string piece = token_to_piece(vocab, tok);
        out += piece;
        if (!piece.empty()) on_delta(piece);

        // 达到最短长度后，遇到 eos 才结束
        if (tok == tok_eos && step >= params.min_new_tokens) {
            break;
        }

        // 2) decode 单 token，并且请求 logits（n_outputs=1）
        llama_batch b1 = llama_batch_init(1, 0, 1);
        b1.n_tokens = 0;
        batch_add(b1, tok, /*pos*/ n_past, /*seq*/ 0, /*logits*/ true);
        n_past++;

        if (llama_decode(ctx, b1) != 0) {
            llama_batch_free(b1);
            break;
        }
        llama_batch_free(b1);
    }

    trim_inplace(out);

    R.ok = true;
    R.text = out;

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return R;
}

} // namespace ws_ai