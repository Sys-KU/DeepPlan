#pragma once
#include <network/session.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
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

template<typename K, typename V>
class LRUCache {
 public:
  bool put(const K& k, const V& v) {
    if(exist(k)) {
      return false;
    }

    items.emplace_front(k, v);

    index.emplace(k, items.begin());
  }

  bool exist(const K& k) {
    return (index.count(k)>0);
  }

  V get(const K& k) {
    assert(exist(k));
    auto itr = index.find(k);

    items.splice(items.begin(), items, itr->second);

    return itr->second->second;
  }

  V pop() {
    auto v = items.back().second;
    index.erase(items.back().first);
    items.pop_back();

    return v;
  }

  size_t size() {
    return index.size();
  }
 private:
  std::list<std::pair<K,V>> items;

  std::unordered_map<K, typename std::list<std::pair<K,V>>::iterator> index;
};

class Worker {
 public:
  Worker(int device);

  void run();

  void infer(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb);

  void init_model(std::vector<std::string> model_names, int n_models,
                  EngineType engine_type, std::vector<int> devices);

  void reset_model();

  void stop();

  at::Device device;

 private:
  size_t capacity_;
  std::atomic_bool alive;
  std::thread worker_thr;
  ModelManager* model_manager = nullptr;
  LRUCache<int, deepplan::Model*>* running_models;
  tbb::concurrent_queue<InferTask> queue_;
};
