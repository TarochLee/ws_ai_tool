#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>
#import <locale.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include "llama.h"
}

// -------------------------
// UTF-8 / string helpers
// -------------------------
static inline void rtrim_inplace(std::string &s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' ||
                        s.back() == '\r' || s.back() == '\t'))
    s.pop_back();
}
static inline void ltrim_inplace(std::string &s) {
  size_t i = 0;
  while (i < s.size() &&
         (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t'))
    i++;
  s.erase(0, i);
}
static inline void trim_inplace(std::string &s) {
  rtrim_inplace(s);
  ltrim_inplace(s);
}

static inline bool contains(const std::string &s, const std::string &sub) {
  return s.find(sub) != std::string::npos;
}

// 把输出截断到 stop 字符串之前（避免模型继续输出额外任务）
static inline void cut_after_any_stop(std::string &s,
                                      const std::vector<std::string> &stops) {
  size_t best = std::string::npos;
  for (const auto &st : stops) {
    size_t p = s.find(st);
    if (p != std::string::npos)
      best = std::min(best, p);
  }
  if (best != std::string::npos)
    s.resize(best);
}

static std::string readFileAll(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    return {};
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

// llama token -> piece (UTF-8 bytes)
static std::string tokenToPiece(const llama_vocab *vocab, llama_token tok) {
  std::string out;
  out.resize(256);
  // llama_token_to_piece(vocab, token, buf, length, lstrip, special)
  int32_t n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                                   /*lstrip*/ 0, /*special*/ true);
  if (n < 0) {
    out.resize((size_t)(-n));
    n = llama_token_to_piece(vocab, tok, out.data(), (int32_t)out.size(),
                             /*lstrip*/ 0, /*special*/ true);
    if (n < 0)
      return {};
  }
  out.resize((size_t)n);
  return out;
}

// -------------------------
// Vision OCR
// -------------------------
static CGImageRef loadCGImage(const std::string &imagePath) {
  NSString *path = [NSString stringWithUTF8String:imagePath.c_str()];
  NSURL *url = [NSURL fileURLWithPath:path];
  CGImageSourceRef src =
      CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
  if (!src)
    return nil;
  CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
  CFRelease(src);
  return img;
}

static std::string ocrWithVision(const std::string &imagePath) {
  CGImageRef img = loadCGImage(imagePath);
  if (!img)
    return {};

  __block NSMutableString *acc = [NSMutableString string];

  VNRecognizeTextRequest *req = [[VNRecognizeTextRequest alloc]
      initWithCompletionHandler:^(VNRequest *request, NSError *error) {
        if (error)
          return;
        NSArray<VNRecognizedTextObservation *> *results =
            (NSArray<VNRecognizedTextObservation *> *)request.results;
        if (![results isKindOfClass:[NSArray class]])
          return;

        for (VNRecognizedTextObservation *obs in results) {
          VNRecognizedText *top = [[obs topCandidates:1] firstObject];
          if (top && top.string.length > 0) {
            [acc appendString:top.string];
            [acc appendString:@"\n"];
          }
        }
      }];

  // 识别设置：中文优先 + 英文兜底
  req.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
  req.usesLanguageCorrection = YES;
  req.minimumTextHeight = 0.012; // 适当降低，截图小字更容易出
  req.recognitionLanguages = @[ @"zh-Hans", @"en-US" ];

  VNImageRequestHandler *handler =
      [[VNImageRequestHandler alloc] initWithCGImage:img options:@{}];
  NSError *err = nil;
  [handler performRequests:@[ req ] error:&err];

  CGImageRelease(img);

  if (err)
    return {};

  std::string out([acc UTF8String] ? [acc UTF8String] : "");
  trim_inplace(out);
  return out;
}

// -------------------------
// llama batch helper (no llama_batch_add)
// -------------------------
static void batch_add_token(llama_batch &batch, llama_token token, int32_t pos,
                            int32_t seq_id, bool logits) {
  const int32_t i = batch.n_tokens;

  batch.token[i] = token;
  batch.pos[i] = pos;
  batch.n_seq_id[i] = 1;
  batch.seq_id[i][0] = seq_id;
  batch.logits[i] = logits ? 1 : 0;

  batch.n_tokens++;
}

// -------------------------
// Build Qwen chat prompt (ChatML)
// -------------------------
static std::string buildPrompt(const std::string &ocrText) {
  // 只输出两段话（不分点），强约束格式，减少跑偏/追加任务
  std::ostringstream oss;
  oss << "<|im_start|>system\n"
      << "你是一个严谨的技术助理。你只使用中文输出，且只输出两段：第一段为对截图文字的总结（不分点，一段话）；第二段为相关扩展知识（不分点，一段话）。不要输出额外标题、不要输出英文、不要输出提问或任务指令。\n"
      << "<|im_end|>\n"
      << "<|im_start|>user\n"
      << "以下是从截图OCR得到的文字，请基于文字内容完成总结与扩展。\n\n"
      << ocrText << "\n<|im_end|>\n"
      << "<|im_start|>assistant\n";
  return oss.str();
}

// tokenize wrapper (new API uses vocab)
static std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                         const std::string &text) {
  int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr,
                             0, /*add_special*/ false, /*parse_special*/ true);
  if (n < 0)
    n = -n;
  std::vector<llama_token> toks(n);
  int32_t n2 = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                              toks.data(), (int32_t)toks.size(),
                              /*add_special*/ false, /*parse_special*/ true);
  if (n2 < 0)
    toks.clear();
  return toks;
}

int main(int argc, char **argv) {
  // 终端输出 UTF-8：确保 locale
  setenv("LC_ALL", "zh_CN.UTF-8", 1);
  setenv("LANG", "zh_CN.UTF-8", 1);
  setlocale(LC_ALL, "");

  std::string imagePath = "pic.jpg";
  std::string modelPath = "models/GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf";

  if (argc >= 2)
    imagePath = argv[1];
  if (argc >= 3)
    modelPath = argv[2];

  // 1) OCR
  std::string ocrText = ocrWithVision(imagePath);
  if (ocrText.empty()) {
    std::cout << "未识别到文字或OCR失败。\n";
    return 0;
  }

  // 2) LLM summarize
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  llama_model *model = llama_model_load_from_file(modelPath.c_str(), mparams);
  if (!model) {
    std::cerr << "模型加载失败: " << modelPath << "\n";
    llama_backend_free();
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 4096;
  cparams.n_batch = 1024;
  llama_context *ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    std::cerr << "上下文创建失败\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  const llama_vocab *vocab = llama_model_get_vocab(model);

  std::string prompt = buildPrompt(ocrText);
  std::vector<llama_token> promptTokens = tokenize(vocab, prompt);
  if (promptTokens.empty()) {
    std::cerr << "prompt tokenize 失败\n";
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // 采样器：温度 + top_p/top_k
  llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(80));
  llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.98f, 1));
  llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.75f));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  // decode prompt（一次性喂进去，logits 只标最后一个 token）
  llama_batch batch = llama_batch_init((int32_t)promptTokens.size(), 0, 1);
  batch.n_tokens = 0;

  for (int32_t i = 0; i < (int32_t)promptTokens.size(); ++i) {
    bool want_logits = (i == (int32_t)promptTokens.size() - 1);
    batch_add_token(batch, promptTokens[i], /*pos*/ i, /*seq*/ 0, want_logits);
  }

  if (llama_decode(ctx, batch) != 0) {
    std::cerr << "llama_decode(prompt) 失败\n";
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // 关键修复：这版 llama.cpp 的 sampler_sample 第三个参数是“batch token 下标”
  // 只有最后一个 token 的 batch.logits[idx] == true
  const int32_t prompt_logits_idx = (int32_t)promptTokens.size() - 1;

  llama_batch_free(batch);

  // stop：遇到 <|im_end|> 或 <|endoftext|> 就停
  const llama_token tok_eos = llama_vocab_eos(vocab);
  std::vector<std::string> stopStrings = {"<|im_end|>", "<|endoftext|>"};

  std::string out;
  out.reserve(8192);

  int32_t n_past = (int32_t)promptTokens.size();

  // 为了避免“越来越短”：设置最小生成长度 + 早 eos 重采样
  const int max_new_tokens = 900;
  const int min_new_tokens = 320;      // 你觉得仍短就调大
  const int max_resample_eos = 96;     // 早 eos 的最多重采样次数

  for (int step = 0; step < max_new_tokens; ++step) {
    // step==0 用 prompt 最后 token 的 logits；后续每轮 batch 只有 1 token，idx=0
    const int32_t sample_idx = (step == 0) ? prompt_logits_idx : 0;

    llama_token tok = 0;

    // 早 EOS：不 accept，重采样几次，避免输出过短
    int eos_resample = 0;
    while (true) {
      tok = llama_sampler_sample(sampler, ctx, sample_idx);

      if (tok == tok_eos && step < min_new_tokens && eos_resample < max_resample_eos) {
        eos_resample++;
        continue;
      }
      break;
    }

    // 如果还是 eos 且已经超过最小长度，就结束
    if (tok == tok_eos && step >= min_new_tokens) {
      break;
    }
    // 如果还是 eos 但还没到最小长度（且重采样用完了），继续也没意义，直接结束
    if (tok == tok_eos && step < min_new_tokens) {
      break;
    }

    llama_sampler_accept(sampler, tok);

    std::string piece = tokenToPiece(vocab, tok);
    out += piece;

    // 若输出中已经出现 stop 字符串，立刻截断并停
    {
      std::string tmp = out;
      cut_after_any_stop(tmp, stopStrings);
      if (tmp.size() != out.size()) {
        out = tmp;
        break;
      }
    }

    // decode 单 token，logits 留给下一轮（此时 token index = 0）
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 0;
    batch_add_token(b, tok, /*pos*/ n_past, /*seq*/ 0, /*logits*/ true);
    n_past += 1;

    if (llama_decode(ctx, b) != 0) {
      llama_batch_free(b);
      break;
    }
    llama_batch_free(b);
  }

  // 最终输出：只打印模型输出（不打印任何调试信息）
  trim_inplace(out);

  // 额外兜底：如果模型仍然输出了多余“任务/问题”，做一次字符串截断
  // 你可以按你观察到的跑偏模板追加 stop 关键字
  std::vector<std::string> extraStops = {
      "Answer the following questions",
      "Generate one question per line",
      "问题：",
      "Questions:",
      "Q1",
      "\n1.",
  };
  cut_after_any_stop(out, extraStops);
  trim_inplace(out);

  std::cout << out << "\n";

  llama_sampler_free(sampler);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}