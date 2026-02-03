#pragma once
#include "ws_ai/config.h"
#include "ws_ai/job_manager.h"
#include <memory>

namespace ws_ai {

class HttpServer {
public:
    HttpServer(Config cfg, std::shared_ptr<JobManager> jm);

    // 阻塞启动
    void serve_forever();

private:
    Config cfg_;
    std::shared_ptr<JobManager> jm_;
};

} // namespace ws_ai