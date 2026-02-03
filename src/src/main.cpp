#include "ws_ai/config.h"
#include "ws_ai/http_server.h"
#include "ws_ai/job_manager.h"

#include <cstdlib>
#include <iostream>
#include <memory>

int main() {
    ws_ai::Config cfg;

    // 环境变量覆盖（可选）
    if (const char *p = std::getenv("WS_AI_PORT")) {
        int v = std::atoi(p);
        if (v > 0 && v < 65536) cfg.port = v;
    }
    if (const char *h = std::getenv("WS_AI_HOST")) {
        if (h && *h) cfg.host = h;   // 现在 Config 有 host 了
    }
    if (const char *m = std::getenv("WS_AI_MODEL")) {
        if (m && *m) cfg.model_path = m;
    }

    auto jm = std::make_shared<ws_ai::JobManager>(cfg);
    ws_ai::HttpServer server(cfg, jm);

    std::cout << "Listening on http://" << cfg.host << ":" << cfg.port << "\n";
    server.serve_forever();
    return 0;
}