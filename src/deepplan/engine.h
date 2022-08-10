#pragma once

#include <c10/cuda/CUDAStream.h>
#include <deepplan/model.h>
#include <util.h>

namespace deepplan {

class Engine {
 public:
  virtual void run(Model* model, ScriptModuleInput& x) = 0;
};

void RunEngine(Model* model, ScriptModuleInput& x);

void Init(void);

void Deinit(void);

}
