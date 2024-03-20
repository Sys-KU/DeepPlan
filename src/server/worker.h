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

struct InferAction {
  InferAction() {};
  InferAction(
    int action_id,
    int model_id,
    std::vector<InferTask> tasks,
    std::function<void(int, uint64_t)> cb)
    : action_id(action_id), model_id(model_id), tasks(tasks), cb(cb) {};

  void complete(const uint64_t exec_time, const bool is_cold) {
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
    cb(action_id, end_time);
  }

  int action_id;
  int model_id;
  std::vector<InferTask> tasks;
  std::function<void(int, uint64_t)> cb;
};


class RequestScoreboard {
 public:
  RequestScoreboard(int num_models, int window_size)
   : scores(num_models, 0),
     window_size_(window_size) {}

  RequestScoreboard(int window_size)
   : window_size_(window_size) {}

  void update_window(int model_id) {
    req_window.push(model_id);

    scores[model_id]++;

    if (req_window.size() > window_size_) {
      scores[req_window.front()]--;
      req_window.pop();
    }
  }

  int get(int model_id) {
    return scores[model_id];
  }

  void clear() {
    while (!req_window.empty()) {
      req_window.pop();
    }
    std::fill(scores.begin(), scores.end(), 0);
  }

  void expand(int num_models) {
    scores.resize(scores.size() + num_models);
    std::fill(scores.begin(), scores.end(), 0);
  }

 private:
  std::queue<int> req_window;

  std::vector<int> scores;

  int window_size_;
};


class CFR {
 public:
  bool put(int model_id, int vruntime) {
    if(exist(model_id)) {
      return false;
    }

    auto item = std::make_pair(vruntime, model_id);

    auto iter = items.begin();
    for (iter; iter != items.end(); iter++) {
      if (iter->first > item.first) {
        auto pos =items.insert(iter, item);
        index.emplace(model_id, pos);
        break;
      }
    }

    if (iter == items.end()) {
      auto pos = items.insert(iter, item);
      index.emplace(model_id, pos);
    }


    return true;
  }

  bool exist(int model_id) {
    return (index.count(model_id)>0);
  }

  std::pair<int, int> pop() {
    auto item = items.front();
    index.erase(item.second);
    items.pop_front();
    return item;
  }

  void erase(int model_id) {
    assert(exist(model_id));
    auto itr = index.find(model_id);

    index.erase(itr);
    items.erase(itr->second);
  }

  size_t size() {
    return index.size();
  }

 private:
  // pair = {vruntime, model_id}
  std::list<std::pair<int, int>> items;

  // key = model_id, valud = iterator of items
  std::unordered_map<int, typename std::list<std::pair<int, int>>::iterator> index;
};


class Worker {
 public:
  Worker(int device, const ServerOptions& options,
         ModelPool* model_pool, std::string name="");
  ~Worker();

  void run();

  void infer(InferAction infer_action);

  void set_r_policy(ReclaimPolicy r_policy);

  libtorch::Model* find_model(int model_id, bool* is_cold);

  void preempt_models();

  void clear_models();

  void sync_setup();

  void stop();

  at::Device device;

  std::string name;

 private:
  const ServerOptions& options_;
  size_t capacity_;
  std::atomic_bool alive;
  std::thread worker_thr;
  ModelPool* model_pool;
  ReclaimPolicy r_policy_;
  util::LRUCache<int, libtorch::Model*>* running_models;
  RequestScoreboard* req_scoreboard = nullptr;
  CFR* cfr;
  std::list<util::LRUCache<int, libtorch::Model*>*> partial_models_list;
  tbb::concurrent_queue<InferAction> queue_;
};
