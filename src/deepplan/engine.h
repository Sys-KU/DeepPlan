#pragma once

#include <c10/cuda/CUDAStream.h>
#include <deepplan/model.h>
#include <util.h>

namespace deepplan {

class Engine {
 public:
  virtual torch::jit::IValue run(Model* model, ScriptModuleInput& x) = 0;
};

torch::jit::IValue RunEngine(Model* model, ScriptModuleInput& x);

void Init(void);

void Deinit(void);

}
