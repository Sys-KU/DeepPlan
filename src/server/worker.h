#pragma once
#include <util.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <deepcache/model.h>
#include <optional>
#include "tbb/concurrent_queue.h"

#define RECLAIM_LAYER_STEP 1
#define RECLAIM_MEMORY_STEP 30 * (1 << 20) // 30 MB
#define MINIMUM_CACHE_MEMORY_RATE 0.3 // 20%

struct InferTask {
  InferTask() {};
  InferTask(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb)
      : request(request),
        cb(cb) {};

  serverapi::InferenceRequest* request;
  std::function<void(serverapi::InferenceResponse*)> cb;
};


class Worker {
 public:
  Worker(int device);
  ~Worker();

  void run();

  void infer(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb);

  void init_model_manager(EngineType engine_type);

  void set_r_policy(ReclaimPolicy r_policy);

  void add_models(std::vector<std::string> model_names, int n_models,
                  EngineType engine_type, std::vector<int> devices);

  libtorch::Model* find_model(int model_id, bool* is_cold);

  void secure_memory_to_load_model(libtorch::Model* model);

  void clear_models();

  void free_models();

  void stop();

  at::Device device;

 private:
  size_t capacity_;
  std::atomic_bool alive;
  std::thread worker_thr;
  ModelManager* model_manager = nullptr;
  ReclaimPolicy r_policy_;
  util::LRUCache<int, libtorch::Model*>* running_models;
  std::list<util::LRUCache<int, libtorch::Model*>*> partial_models_list;
  tbb::concurrent_queue<InferTask> queue_;
};
