#pragma once

#include "ws_ai/config.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory> // ✅ 如果你头里用 shared_ptr / unique_ptr
#include <mutex>
#include <queue>
#include <string>
#include <thread> // ✅ 必须：std::thread
#include <unordered_map>

namespace ws_ai {
class Pipeline;

enum class JobState { queued, running, done, error };

struct JobInfo {
  std::string id;
  std::string image_path;

  JobState state = JobState::queued;
  int progress = 0;   // 0..100（给前端进度条）
  std::string result; // done 时填
  std::string error;  // error 时填

  std::chrono::system_clock::time_point created_at;
};

// OCR + LLM 流水线接口（避免在头文件里引入 Objective-C / Vision）
// class Pipeline {
// public:
//     virtual ~Pipeline() = default;
//     virtual std::string run(const std::string &image_path,
//                             std::atomic<int> &progress,
//                             std::atomic<bool> &cancel_flag,
//                             std::string &err_out) = 0;
// };

// 工厂函数：在 pipeline.mm 里实现
std::unique_ptr<Pipeline> make_pipeline(const Config &cfg);

class JobManager : public std::enable_shared_from_this<JobManager> {
public:
  explicit JobManager(Config cfg);
  ~JobManager();

  // http_server.cpp 需要的两个接口：
  std::string submit_image(const std::string &image_path);
  std::string get_status_json(const std::string &id) const;

private:
  void worker_loop();
  std::string new_id() const;
  static const char *state_to_cstr(JobState s);
  static std::string json_escape(const std::string &s);

private:
  Config cfg_;

  std::unique_ptr<Pipeline> pipeline_;

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;

  // 存所有 job 的信息（查询用）
  std::unordered_map<std::string, JobInfo> jobs_;

  // 排队等待执行的 job id
  std::queue<std::string> queue_;

  // worker
  std::thread worker_;
  std::atomic<bool> stop_{false};

  // 当前任务进度（worker 写，status 读）
  // 为简单起见：每个任务执行时用一个局部 atomic，再把值拷回 jobs_。
};

} // namespace ws_ai