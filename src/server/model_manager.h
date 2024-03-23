#pragma once

#include <deepplan/model.h>
#include <util.h>

size_t getDeviceActiveMemorySize(int deivce);

struct ReclaimingOutput {
  int model_id;
  size_t size;
  std::vector<int> layers;
};


class ModelPool {
 public:
  ModelPool(std::string model_repo)
    : model_repo(model_repo) {};

  int get_num_models() {
    return models.size();
  }

  deepplan::Model* get_model(int model_id) {
    return models[model_id];
  }

  ReclaimingOutput reclaim_model(int model_id, size_t size=SIZE_MAX) {
    auto [reclaimed_size, r_layers] = models[model_id]->reclaim_memory(size);
    ReclaimingOutput output = {model_id, reclaimed_size, r_layers};
    return output;
  }

  uint64_t get_model_exec_time(int model_id, int batch_size);

  void add_model(std::string model_name, EngineType engine_type,
                 std::vector<int> devices);

  void reset_models() {
    while (!models.empty()) {
      delete models.back();
      models.pop_back();
    }
  }

  std::string model_repo;

  std::vector<deepplan::Model*> models;
};

class ModelManager {
 public:
  ModelManager(std::string model_repo, int n_devices)
    : model_repo(model_repo) {
    model_pools.resize(n_devices);
    for (int i = 0; i < model_pools.size(); i++) {
      model_pools[i] = new ModelPool(model_repo);
    }
  };

  ~ModelManager() {
    for (auto pool : model_pools) {
      pool->reset_models();
    }
  }

  bool setup(std::vector<std::string> model_names, int num_models,
             EngineType engine_type, int mp_size);

  std::vector<ModelPool*> model_pools;

  std::string model_repo;

 private:
  std::vector<std::string> model_names_ = {""};
  int num_models_ = 0;
  EngineType engine_type_ = EngineType::PIPESWITCH;
  int mp_size_ = 0;
};
