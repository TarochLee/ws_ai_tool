#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>

#include <string>
#include <vector>
#include <iostream>

extern "C" {
#include "llama.h"
}

// -------------------- OCR (macOS Vision) --------------------
static CGImageRef load_cgimage_from_path(NSString *path) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!src) return nullptr;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    return img; // caller must CGImageRelease
}

static std::string ocr_vision(const std::string &image_path_utf8) {
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:image_path_utf8.c_str()];
        CGImageRef img = load_cgimage_from_path(path);
        if (!img) return "";

        __block NSMutableArray<NSString *> *lines = [NSMutableArray array];

        VNRecognizeTextRequest *req =
        [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
            if (error) return;
            for (VNRecognizedTextObservation *obs in request.results) {
                VNRecognizedText *top = [[obs topCandidates:1] firstObject];
                if (top && top.string.length > 0) {
                    [lines addObject:top.string];
                }
            }
        }];

        req.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        req.usesLanguageCorrection = YES;
        req.recognitionLanguages = @[@"zh-Hans", @"en-US"];

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:img options:@{}];
        NSError *err = nil;
        BOOL ok = [handler performRequests:@[req] error:&err];

        CGImageRelease(img);

        if (!ok || err) return "";

        NSMutableString *out = [NSMutableString string];
        for (NSString *ln in lines) {
            if (ln.length == 0) continue;
            [out appendString:ln];
            [out appendString:@"\n"];
        }
        return std::string([out UTF8String]);
    }
}

// -------------------- llama.cpp helpers (new API) --------------------
static const llama_vocab * get_vocab(llama_model * model) {
    return llama_model_get_vocab(model);
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(
        vocab,
        text.c_str(),
        (int32_t)text.size(),
        tokens.data(),
        (int32_t)tokens.size(),
        /* add_special */ true,
        /* parse_special */ true
    );
    if (n < 0) return {};
    tokens.resize((size_t)n);
    return tokens;
}

static int decode_tokens(llama_context *ctx, const std::vector<llama_token> &toks, int &n_past) {
    llama_batch batch = llama_batch_init((int32_t)toks.size(), 0, 1);

    batch.n_tokens = (int32_t)toks.size();
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i]     = toks[(size_t)i];
        batch.pos[i]       = n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == batch.n_tokens - 1);
    }

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc == 0) n_past += (int)toks.size();
    return rc;
}

static int decode_one(llama_context *ctx, llama_token tok, int &n_past) {
    llama_batch batch = llama_batch_init(1, 0, 1);

    batch.n_tokens = 1;
    batch.token[0]     = tok;
    batch.pos[0]       = n_past;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc == 0) n_past += 1;
    return rc;
}

// 解决“ãĢĲ...”乱码：用 llama_token_to_piece 正确还原 UTF-8
static std::string token_to_utf8_piece(const llama_vocab *vocab, llama_token tok) {
    std::string buf;
    buf.resize(4096);

    // 你这版签名：vocab, token, buf, length, lstrip, special
    int32_t n = llama_token_to_piece(
        vocab,
        tok,
        buf.data(),
        (int32_t)buf.size(),
        /*lstrip*/ 0,
        /*special*/ true
    );

    if (n < 0) return "";
    buf.resize((size_t)n);
    return buf;
}

static std::string generate_text(
    const std::string &model_path,
    const std::string &prompt,
    int n_ctx = 4096,
    int n_predict = 900,
    float temperature = 0.85f,
    int top_k = 40,
    float top_p = 0.9f
) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        llama_backend_free();
        return "模型加载失败：请检查 GGUF 路径。\n";
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx;
    cparams.n_batch = 1024;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        return "上下文创建失败。\n";
    }

    const llama_vocab *vocab = get_vocab(model);

    auto ptoks = tokenize(vocab, prompt);
    if (ptoks.empty()) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return "tokenize 失败。\n";
    }

    int n_past = 0;
    if (decode_tokens(ctx, ptoks, n_past) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return "prompt decode 失败。\n";
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_token eos = llama_vocab_eos(vocab);

    std::string out;
    out.reserve(8192);

    for (int i = 0; i < n_predict; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (tok == eos) break;

        out += token_to_utf8_piece(vocab, tok);

        if (decode_one(ctx, tok, n_past) != 0) break;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return out;
}

// -------------------- main --------------------
int main() {
    const std::string image_path = "pic.jpg";
    const std::string model_path = "models/GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    std::string ocr = ocr_vision(image_path);
    if (ocr.size() < 30) {
        std::cout << "OCR文本过少，无法生成结果。\n";
        return 0;
    }

    const std::string prompt =
        "你将收到一段从截图OCR提取的文本，可能有噪声或不完整。\n"
        "请生成专业、详细的技术笔记，并扩展相关知识点。\n"
        "要求：\n"
        "1) 【客观摘要】较长段落复述核心内容；\n"
        "2) 【扩展讲解】补充背景、常见误区、工程注意点；\n"
        "3) 【使用者意图推测】用“可能/推测/也许”写3-6条，并在每条后附OCR证据短语；\n"
        "4) 输出只用中文和常见标点；\n"
        "——OCR文本开始——\n" + ocr +
        "——OCR文本结束——\n";

    std::string res = generate_text(model_path, prompt, 4096, 1200, 0.85f, 40, 0.9f);
    std::cout << res << "\n";
    return 0;
}