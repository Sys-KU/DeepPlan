#pragma once

#include <torch/script.h>
#include <util.h>
#include <map>
#include <deepcache.pb.h>

struct InputConfig {
 public:
  InputConfig(ModelInput io)
  : shape(io.shape().begin(), io.shape().end()),
    data_type(io.data_type()) {}

  std::vector<int64_t> shape;
  DataType data_type;
};

struct Model {
 public:
  Model(const std::string name, const EngineType type, const std::vector<int> devices);

  Model() {};

  void init();

  void forward(ScriptModuleInput& x);

  void to(at::Device device, bool non_blocking = false);

  void clear();

  std::string model_name;

  EngineType engine_type;

  std::vector<int> devices;

  ScriptModule model;

  size_t model_size;

  std::vector<InputConfig> inputs;

  std::vector<ScriptModule> layers;

  std::map<int, std::vector<int>> device_map;

  int n_layers;

  std::vector<int> load_layer_idxs;

  std::atomic<bool> is_cuda;

  ModelConfig model_config;
};

