#pragma once
#include <atomic>
#include <memory>
#include <string>
#include "ws_ai/config.h"

namespace ws_ai {

class Pipeline {
public:
    virtual ~Pipeline() = default;
    virtual std::string run(const std::string &image_path,
                            std::atomic<int> &progress,
                            std::atomic<bool> &cancel_flag,
                            std::string &err_out) = 0;
};

std::unique_ptr<Pipeline> make_pipeline(const Config &cfg);

} // namespace ws_ai