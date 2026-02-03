#pragma once
#include <string>

namespace ws_ai {

struct Config {
  // http server
  std::string host = "0.0.0.0";  // 新增：监听地址（默认对外）
  int port = 8080;

  // paths
  std::string model_path = "models/GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf";
  std::string web_root   = "src/web";
  std::string upload_dir = "/tmp/ws_ai_upload";
  std::string job_dir    = "/tmp/ws_ai_jobs";

  // llama context
  int n_ctx   = 4096;
  int n_batch = 1024;

  // sampling
  int   top_k = 40;
  float top_p = 0.90f;
  float temp  = 0.40f;

  int max_new_tokens   = 800;
  int min_new_tokens   = 160;
  int max_resample_eos = 64;
};

} // namespace ws_ai