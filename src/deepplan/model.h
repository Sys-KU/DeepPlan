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

struct LoadState {
  LoadState(int idx, Device device, size_t size, uint64_t load_time)
    : idx(idx), device(device), size(size), load_time(load_time) {}
  int idx;
  Device device;
  size_t size;
  uint64_t load_time;
};

class ModelInstance {
 public:
   ModelInstance(const std::string script_path,
                 const std::vector<InputConfig> input_configs,
                 const at::Device device,
                 const std::vector<int> load_layers);

  void init(const std::string script_path, const std::vector<int> load_layers);

  torch::jit::IValue forward(ScriptModuleInput& x,
                             const std::vector<int> load_layer_idxs);

  void reclaim_layers(std::vector<int> reclaiming_layers);

  void load_layers(std::vector<int> load_layers, bool non_blocking);

  void clear() {
    model.clear();
  }

 private:
  at::Device target_device;

  ScriptModule model;

  std::vector<ScriptModule> layers;

  std::vector<InputConfig> input_configs;
};


class Model : public libtorch::Model {
 public:
  Model(const std::string name, const std::string model_path, const EngineType type, const std::vector<int> devices);
  ~Model() {
    if (model_instance != nullptr) {
      delete model_instance;
    }
  };

  void init();

	torch::jit::IValue forward(ScriptModuleInput& x);

  void to(at::Device device, bool non_blocking = false);

  void clear();

  std::pair<size_t, std::vector<int>> reclaim_memory(size_t size);

  void reclaim_memory(double rate);

  std::pair<size_t, std::vector<int>> load_layers(bool non_blocking = false);

  std::pair<size_t, std::vector<int>> load_layers(int n_layers, bool non_blocking = false);

  uint64_t get_load_time();

  uint64_t get_model_exec_time(int batch_size);

  std::vector<int> get_load_layers() {
    std::vector<int> load_layers;
    for (auto& load_state : load_state_maps) {
      if (load_state.device == Device::CPU) {
        load_layers.push_back(load_state.idx);
      }
    }

    return std::move(load_layers);
  }

  ModelInstance* model_instance;

  // load_state_maps represent the load state whether layer is loaded or not.
  std::vector<LoadState> load_state_maps;

  // uncached_size represent the unloaded size of layers
  // that are required to execute this model
  size_t uncached_size;

  EngineType engine_type;

  std::vector<int> devices = {0};

  std::unordered_map<int, std::vector<int>> device_map;

  size_t optimal_size;
  uint32_t optimal_idx;

  Prof prof;
};

}
