#pragma once

#include <c10/cuda/CUDAStream.h>
#include <util.h>

namespace engine {

class Engine {
 public:
  virtual void run(Model* model, ScriptModuleInput& x) = 0;
};

void run(Model* model, ScriptModuleInput& x);

void init(void);

void deinit(void);

}
