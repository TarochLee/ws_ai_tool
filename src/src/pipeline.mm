#import <Foundation/Foundation.h>
#import <locale.h>

#include "ws_ai/pipeline.h"
#include "ws_ai/config.h"
#include "ws_ai/ocr_vision.h"   // 正确函数：ocr_with_vision
#include "ws_ai/prompt.h"       // 正确函数：build_prompt
#include "ws_ai/util.h"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include "llama.h"
}

namespace ws_ai {

// -------------------------
// string helpers
// -------------------------
static inline void rtrim_inplace(std::string &s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' ||
                        s.back() == '\r' || s.back() == '\t')) {
    s.pop_back();
  }
}
static inline void ltrim_inplace(std::string &s) {
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' ||
                          s[i] == '\r' || s[i] == '\t')) {
    i++;
  }
  s.erase(0, i);
}
static inline void trim_inplace(std::string &s) {
  rtrim_inplace(s);
  ltrim_inplace(s);
}

// 把输出截断到 stop 字符串之前
static inline void cut_after_any_stop(std::string &s,
                                      const std::vector<std::string> &stops) {
  size_t best = std::string::npos;
  for (const auto &st : stops) {
    size_t p = s.find(st);
    if (p != std::string::npos) best = std::min(best, p);
  }
  if (best != std::string::npos) s.resize(best);
}

// llama token -> piece
static std::string token_to_piece(const llama_vocab *vocab, llama_token tok) {
  std::string out;
  out.resize(256);
  int32_t n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                                   /*lstrip*/0, /*special*/true);
  if (n < 0) {
    out.resize((size_t)(-n));
    n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                             /*lstrip*/0, /*special*/true);
    if (n < 0) return {};
  }
  out.resize((size_t)n);
  return out;
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                         const std::string &text) {
  int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                             nullptr, 0,
                             /*add_special*/false,
                             /*parse_special*/true);
  if (n < 0) n = -n;
  std::vector<llama_token> toks(n);
  int32_t n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                              toks.data(), (int32_t)toks.size(),
                              /*add_special*/false,
                              /*parse_special*/true);
  if (n2 < 0) toks.clear();
  return toks;
}

// -------------------------
// llama batch helper (不用 llama_batch_add，兼容你当前版本)
// -------------------------
static void batch_add_token(llama_batch &batch, llama_token token, int32_t pos,
                            int32_t seq_id, bool logits) {
  const int32_t i = batch.n_tokens;
  batch.token[i]    = token;
  batch.pos[i]      = pos;
  batch.n_seq_id[i] = 1;
  batch.seq_id[i][0]= seq_id;
  batch.logits[i]   = logits ? 1 : 0;
  batch.n_tokens++;
}

// -------------------------
// Pipeline impl
// -------------------------
class PipelineImpl final : public Pipeline {
public:
  explicit PipelineImpl(const Config &cfg) : cfg_(cfg) {}

  // 必须和 pipeline.h 完全一致：run(image_path, progress, cancel_flag, err_out)
  std::string run(const std::string &image_path,
                  std::atomic<int> &progress,
                  std::atomic<bool> &cancel_flag,
                  std::string &err_out) override {
    err_out.clear();
    progress.store(1);

    // locale（避免中文乱码）
    setenv("LC_ALL", "zh_CN.UTF-8", 1);
    setenv("LANG", "zh_CN.UTF-8", 1);
    setlocale(LC_ALL, "");

    if (cancel_flag.load()) {
      err_out = "cancelled";
      return "";
    }

    // 1) OCR
    std::string ocr = ocr_with_vision(image_path);
    trim_inplace(ocr);
    if (ocr.empty()) {
      err_out = "OCR失败或未识别到文字";
      progress.store(100);
      return "";
    }
    progress.store(10);

    if (cancel_flag.load()) {
      err_out = "cancelled";
      return "";
    }

    // 2) prompt（用你 prompt.cpp 提供的 build_prompt）
    const std::string prompt = build_prompt(ocr);
    progress.store(15);

    // 3) llama init
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(cfg_.model_path.c_str(), mparams);
    if (!model) {
      err_out = "模型加载失败: " + cfg_.model_path;
      llama_backend_free();
      progress.store(100);
      return "";
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = cfg_.n_ctx;
    cparams.n_batch = cfg_.n_batch;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
      err_out = "llama context 创建失败";
      llama_model_free(model);
      llama_backend_free();
      progress.store(100);
      return "";
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // tokenize prompt
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
    if (prompt_tokens.empty()) {
      err_out = "prompt tokenize 失败";
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      progress.store(100);
      return "";
    }

    // 4) sampler
    llama_sampler *sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(cfg_.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(cfg_.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(cfg_.temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 5) decode prompt（关键：只给最后一个 token 打 logits=1）
    llama_batch batch = llama_batch_init((int32_t)prompt_tokens.size(), 0, 1);
    batch.n_tokens = 0;
    for (int32_t i = 0; i < (int32_t)prompt_tokens.size(); ++i) {
      const bool want_logits = (i == (int32_t)prompt_tokens.size() - 1);
      batch_add_token(batch, prompt_tokens[i], /*pos*/i, /*seq*/0, want_logits);
    }

    if (llama_decode(ctx, batch) != 0) {
      err_out = "llama_decode(prompt) 失败";
      llama_batch_free(batch);
      llama_sampler_free(sampler);
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      progress.store(100);
      return "";
    }

    // 关键修复：第一次采样必须用 prompt 最后 token 的 logits index（n_prompt-1）
    const int32_t prompt_logits_idx = (int32_t)prompt_tokens.size() - 1;

    llama_batch_free(batch);

    // 6) generation
    const llama_token tok_eos = llama_vocab_eos(vocab);
    const std::vector<std::string> stopStrings = {"<|im_end|>", "<|endoftext|>"};

    std::string out;
    out.reserve(8192);

    int32_t n_past = (int32_t)prompt_tokens.size();
    int eos_resample_left = cfg_.max_resample_eos;

    for (int step = 0; step < cfg_.max_new_tokens; ++step) {
      if (cancel_flag.load()) {
        err_out = "cancelled";
        break;
      }

      // 进度条：15% ~ 95%
      const int p = 15 + (int)((double)step / std::max(1, cfg_.max_new_tokens) * 80.0);
      progress.store(std::min(95, p));

      // step==0 用 prompt_logits_idx；之后每轮 decode 的 batch 只有 1 token，logits index 就是 0
      int32_t sample_idx = (step == 0) ? prompt_logits_idx : 0;

      llama_token tok = llama_sampler_sample(sampler, ctx, sample_idx);

      // 早 eos：在 min_new_tokens 前尽量重采样，避免“越来越短”
      if (tok == tok_eos && step < cfg_.min_new_tokens && eos_resample_left > 0) {
        int tries = std::min(eos_resample_left, 8);
        bool replaced = false;
        while (tries-- > 0) {
          llama_token t2 = llama_sampler_sample(sampler, ctx, sample_idx);
          if (t2 != tok_eos) {
            tok = t2;
            replaced = true;
            break;
          }
        }
        eos_resample_left--;
        if (!replaced && eos_resample_left > 0) {
          // 继续下一轮再试一次（不 accept eos）
          step--;
          continue;
        }
      }

      // 如果允许结束
      if (tok == tok_eos && step >= cfg_.min_new_tokens) {
        break;
      }

      // accept
      llama_sampler_accept(sampler, tok);

      // append text
      out += token_to_piece(vocab, tok);

      // stop strings
      std::string tmp = out;
      cut_after_any_stop(tmp, stopStrings);
      if (tmp.size() != out.size()) {
        out = tmp;
        break;
      }

      // decode this token（logits=1 留给下一轮采样 idx=0）
      llama_batch b = llama_batch_init(1, 0, 1);
      b.n_tokens = 0;
      batch_add_token(b, tok, /*pos*/n_past, /*seq*/0, /*logits*/true);
      n_past += 1;

      if (llama_decode(ctx, b) != 0) {
        llama_batch_free(b);
        err_out = "llama_decode(next) 失败";
        break;
      }
      llama_batch_free(b);
    }

    trim_inplace(out);
    if (!err_out.empty() && out.empty()) {
      // cancelled 或 decode 失败时可能为空
    }

    progress.store(100);

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return out;
  }

private:
  Config cfg_;
};

// 工厂函数：提供给 JobManager 调用（必须有定义，否则会链接失败）
std::unique_ptr<Pipeline> make_pipeline(const Config &cfg) {
  return std::make_unique<PipelineImpl>(cfg);
}

} // namespace ws_ai