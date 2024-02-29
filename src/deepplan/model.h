#pragma once

#include <torch/script.h>
#include <util.h>
#include <unordered_map>
#include <deepplan.pb.h>

#include <libtorch/model.h>

namespace deepplan {

typedef enum {
  CPU = 0,
  CUDA,
} Device;

class Model : public libtorch::Model {
 public:
  Model(const std::string name, const std::string model_path, const EngineType type, const std::vector<int> devices);

  void init();

	torch::jit::IValue forward(ScriptModuleInput& x);

  void to(at::Device device, bool non_blocking = false);

  void clear();

  void reclaim_layers(int n_layers);

  void reclaim_memory(size_t size);

  void reclaim_memory(double rate);

  void load_layers(bool non_blocking = false);

  void load_layers(int n_layers, bool non_blocking = false);

  // load_state_maps represent the load state whether layer is loaded or not.
  std::vector<std::pair<int, Device>> load_state_maps;

  // uncached_size represent the unloaded size of layers
  // that are required to execute this model
  size_t uncached_size;

  EngineType engine_type;

  std::vector<int> devices = {0};

  std::unordered_map<int, std::vector<int>> device_map;
};

}
