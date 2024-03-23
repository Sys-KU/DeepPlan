#pragma once

#include <torch/script.h>
#include <util.h>
#include <deepplan.pb.h>

namespace libtorch {

class Model {
 public:
  Model(const std::string name, const std::string model_path, const int device=0);

  void init(const std::string model_path);

	virtual torch::jit::IValue forward(ScriptModuleInput& x);

  virtual void to(at::Device device, bool non_blocking = false);

  virtual void clear();

  std::string model_name;

  at::Device target_device;

  ScriptModule model;

  size_t model_size;

  int n_layers;

  std::vector<InputConfig> input_configs;

  std::string script_path;
  std::string config_path;

  ModelConfig model_config;
};

}
