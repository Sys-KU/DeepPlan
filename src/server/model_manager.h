#pragma once

#include <deepplan/model.h>
#include <deepcache/model.h>
#include <util.h>

size_t getDeviceActiveMemorySize(int deivce);

class ModelManager {
 public:
  ModelManager(EngineType engine_type)
    : engine_type(engine_type) {};

  ~ModelManager() {
    for (auto &model : models) {
      delete model;
    }
  }

  void add_model(std::string model_name, std::vector<int> devices);

  libtorch::Model* get_model(int model_id) {
    return models[model_id];
  }

  void clear();

  EngineType engine_type;

 private:
  std::vector<libtorch::Model*> models;
};
