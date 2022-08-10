#pragma once

#include <deepplan/model.h>
#include <util.h>

size_t getDeviceActiveMemorySize(int deivce);

class ModelManager {
 public:
  ModelManager(EngineType engine_type)
    : engine_type(engine_type) {};

  void add_model(std::string model_name, std::vector<int> devices);

  deepplan::Model* get_model(int model_id);

  void clear();

  EngineType engine_type;

 private:
  std::vector<deepplan::Model*> models;
};
