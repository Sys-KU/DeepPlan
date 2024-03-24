#pragma once
#include <util.h>
#include <time_util.h>
#include <options.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <optional>
#include <queue>
#include "tbb/concurrent_queue.h"

#define RECLAIM_LAYER_STEP 1
#define RECLAIM_MEMORY_STEP 30 * (1 << 20) // 30 MB
#define RECLAIM_MEMORY_RATE 0.1 // 10%
#define MINIMUM_CACHE_MEMORY_RATE 0.3 // 30%
#define MAX_WINDOW_SIZE 5000

struct InferTask {
  InferTask() {};
  InferTask(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb,
      std::function<void(serverapi::TimeoutResponse*)> timeout_cb)
      : request(request),
        cb(cb),
        timeout_cb(timeout_cb) {};

  serverapi::InferenceRequest* request;
  std::function<void(serverapi::InferenceResponse*)> cb;
  std::function<void(serverapi::TimeoutResponse*)> timeout_cb;

  bool operator<(const InferTask& other) const {
    return request->deadline < other.request->deadline;
  }
};

struct Action {
  Action() {};
  Action(int action_id)
    : action_id(action_id) {};

  virtual ~Action() {};

  int action_id;
};

struct InferAction : public Action {
  InferAction() {};
  InferAction(
    int action_id,
    int model_id,
    std::vector<int> layers,
    std::vector<InferTask> tasks,
    std::function<void(int, uint64_t, uint64_t)> cb)
    : Action(action_id), model_id(model_id), layers(layers), tasks(tasks), cb(cb) {};

  void complete(const uint64_t exec_time, const uint64_t end_size, const bool is_cold) {
    uint64_t end_time = util::now();
    for (const auto& task : tasks) {
      auto response = new serverapi::InferenceResponse();
      response->req_id = task.request->req_id;
      response->is_cold = is_cold;
      response->infer_time = exec_time;
      response->arrival_time = task.request->arrival_time;
      response->deadline = task.request->deadline;
      response->response_time = end_time;
      task.cb(response);
    }
    cb(action_id, end_time, end_size);
  }

  int model_id;
  std::vector<int> layers;
  std::vector<InferTask> tasks;
  std::function<void(int, uint64_t, uint64_t)> cb;
};


struct ReclaimAction : public Action {
  ReclaimAction() {};
  ReclaimAction(
    int action_id,
    std::vector<ReclaimingOutput> outputs,
    std::function<void(int, uint64_t)> cb)
    : Action(action_id), outputs(outputs), cb(cb) {};

  void complete(const uint64_t end_size) {
    cb(action_id, end_size);
  }

  std::vector<ReclaimingOutput> outputs;
  std::function<void(int, uint64_t)> cb;
};


using namespace deepplan;

class Worker {
 public:
  Worker(int device, const ServerOptions& options,
         std::string name="");
  ~Worker();

  void run();

  void infer(InferAction infer_action);

  void reclaim(ReclaimAction reclaim_action);

  void clear_models();

  void sync_setup(std::vector<ModelInstance*> model_instances);

  void stop();

  at::Device device;

  std::string name;

 private:
  const ServerOptions& options_;
  std::atomic_bool alive;
  std::thread worker_thr;
  std::vector<ModelInstance*> model_instances;
  tbb::concurrent_queue<std::shared_ptr<Action>> queue_;
};
