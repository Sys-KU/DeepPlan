#pragma once
#include <util.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <deepcache/model.h>
#include <optional>
#include "tbb/concurrent_queue.h"

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

  void run();

  void infer(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb);

  void init_model_manager(EngineType engine_type);

  void add_models(std::vector<std::string> model_names, int n_models,
                  EngineType engine_type, std::vector<int> devices);

  void clear_models();

  void free_models();

  void stop();

  at::Device device;

 private:
  size_t capacity_;
  std::atomic_bool alive;
  std::thread worker_thr;
  ModelManager* model_manager = nullptr;
  util::LRUCache<int, libtorch::Model*>* running_models;
  util::LRUCache<int, libtorch::Model*>* partial_models;
  tbb::concurrent_queue<InferTask> queue_;
};
