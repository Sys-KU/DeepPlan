#pragma once

#include <c10/cuda/CUDAStream.h>
#include <deepplan/model.h>
#include <util.h>

namespace deepplan {

class Engine {
 public:
  virtual torch::jit::IValue run(
      ScriptModule& model,
      ScriptModuleInput& x,
      int target_device,
      std::unordered_map<int, std::vector<ScriptModule>>& device_map) = 0;
};

torch::jit::IValue RunEngine(
    ScriptModule& model,
    ScriptModuleInput& x,
    at::Device target_device,
    std::unordered_map<int, std::vector<ScriptModule>>& device_map);

void LoadLayers(
    at::Device target_device,
    std::unordered_map<int, std::vector<ScriptModule>>& device_map);

void Init(void);

void Deinit(void);

}
