#include "ws_ai/job_manager.h"
#include "ws_ai/pipeline.h" // 只放一个薄头，里面声明 make_pipeline

#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

namespace ws_ai {

// 把 Pipeline 工厂声明放到单独头里更干净；这里直接 include 也行
// std::unique_ptr<Pipeline> make_pipeline(const Config &cfg);

JobManager::JobManager(Config cfg) : cfg_(std::move(cfg)) {
    // 先做单 worker，稳定；需要并发再扩成线程池
    worker_ = std::thread([this] { worker_loop(); });
}

JobManager::~JobManager() {
    stop_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::string JobManager::new_id() const {
    // 简单可用：时间戳 + 随机数（够用）
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    uint64_t r = rng();
    uint64_t t = (uint64_t)std::time(nullptr);

    std::ostringstream oss;
    oss << std::hex << t << "_" << r;
    return oss.str();
}

const char *JobManager::state_to_cstr(JobState s) {
    switch (s) {
        case JobState::queued:  return "queued";
        case JobState::running: return "running";
        case JobState::done:    return "done";
        case JobState::error:   return "error";
    }
    return "unknown";
}

std::string JobManager::json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '\"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (int)c);
                    o += buf;
                } else {
                    o.push_back((char)c);
                }
        }
    }
    return o;
}

std::string JobManager::submit_image(const std::string &image_path) {
    JobInfo job;
    job.id = new_id();
    job.image_path = image_path;
    job.state = JobState::queued;
    job.progress = 0;
    job.created_at = std::chrono::system_clock::now();

    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.emplace(job.id, job);
        queue_.push(job.id);
    }
    cv_.notify_one();
    return job.id;
}

std::string JobManager::get_status_json(const std::string &id) const {
    JobInfo job;
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = jobs_.find(id);
        if (it != jobs_.end()) {
            job = it->second; // 拷贝一份，避免锁持有太久
            found = true;
        }
    }

    if (!found) {
        return "{\"ok\":false,\"error\":\"not found\"}";
    }

    std::ostringstream oss;
    oss << "{"
        << "\"ok\":true,"
        << "\"id\":\"" << json_escape(job.id) << "\","
        << "\"state\":\"" << state_to_cstr(job.state) << "\","
        << "\"progress\":" << job.progress << ",";

    if (job.state == JobState::done) {
        oss << "\"result\":\"" << json_escape(job.result) << "\"";
    } else if (job.state == JobState::error) {
        oss << "\"error\":\"" << json_escape(job.error) << "\"";
    } else {
        oss << "\"result\":\"\"";
    }
    oss << "}";
    return oss.str();
}

void JobManager::worker_loop() {
    // 每次循环只取一个任务执行
    while (!stop_.load()) {
        std::string id;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_.load() || !queue_.empty(); });
            if (stop_.load()) break;
            id = queue_.front();
            queue_.pop();

            auto it = jobs_.find(id);
            if (it == jobs_.end()) continue;
            it->second.state = JobState::running;
            it->second.progress = 5;
        }

        // 任务执行（不持锁）
        std::atomic<int> progress{5};
        std::atomic<bool> cancel{false};
        std::string err;

        std::unique_ptr<Pipeline> pipe = make_pipeline(cfg_);
        std::string result = pipe->run(
            /*image_path*/[&]{
                std::lock_guard<std::mutex> lk(mu_);
                return jobs_[id].image_path;
            }(),
            progress, cancel, err
        );

        // 写回结果
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = jobs_.find(id);
            if (it == jobs_.end()) continue;

            // 把 pipeline 的 progress 写回 job
            it->second.progress = std::max(0, std::min(100, progress.load()));

            if (!err.empty()) {
                it->second.state = JobState::error;
                it->second.error = err;
                it->second.progress = 100;
            } else {
                it->second.state = JobState::done;
                it->second.result = result;
                it->second.progress = 100;
            }
        }
    }
}

} // namespace ws_ai