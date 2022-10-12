#pragma once

#include <c10/cuda/CUDAStream.h>
#include <deepcache/model.h>
#include <util.h>

namespace deepcache {

class Engine {
 public:
  virtual torch::jit::IValue run(Model* model, ScriptModuleInput& x) = 0;
};

torch::jit::IValue RunEngine(Model* model, ScriptModuleInput& x);

void Init(void);

void Deinit(void);

}
