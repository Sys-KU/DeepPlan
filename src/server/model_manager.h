#pragma once

#include <deepplan/model.h>
#include <util.h>

size_t getDeviceActiveMemorySize(int deivce);

class ModelPool {
 public:
  ModelPool(std::string model_repo)
    : model_repo(model_repo) {};

  int get_num_models() {
    return models.size();
  }

  libtorch::Model* get_model(int model_id) {
    return models[model_id];
  }

  void add_model(std::string model_name, EngineType engine_type,
                 std::vector<int> devices);

  void reset_models() {
    for (auto model : models) {
      delete model;
    }
  }

  std::string model_repo;

 private:
  std::vector<libtorch::Model*> models;
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
