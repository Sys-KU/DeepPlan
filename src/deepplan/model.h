#pragma once

#include <torch/script.h>
#include <util.h>
#include <unordered_map>
#include <deepplan.pb.h>

#include <libtorch/model.h>

namespace deepplan {

class Model : public libtorch::Model {
 public:
  Model(const std::string name, const std::string model_path, const EngineType type, const std::vector<int> devices);

  void init();

	torch::jit::IValue forward(ScriptModuleInput& x);

  void to(at::Device device, bool non_blocking = false);

  void clear();

  EngineType engine_type;

  std::vector<int> devices = {0};

  std::unordered_map<int, std::vector<int>> device_map;

  std::vector<int> load_layer_idxs;
};

}
