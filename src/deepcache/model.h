#pragma once

#include <torch/script.h>
#include <util.h>
#include <deepplan.pb.h>

#include <libtorch/model.h>

namespace deepcache {

typedef enum {
  CPU = 0,
  CUDA,
} Device;

class Model : public libtorch::Model {
 public:
   Model(const std::string name, const std::string model_path, const int device);

   void init();

   torch::jit::IValue forward(ScriptModuleInput& x);

   void to(at::Device device, bool non_blocking = false);

   void clear();

   void reclaim_layers(int n_layers);

   void reclaim_memory(size_t size);

   void load_layers(int n_layers, bool non_blocking = false);

   std::vector<int> get_host_layers();

   // layers_load_info represent the load info whether layer is loaded or not.
   std::vector<Device> layers_load_info;

   // remained_size represent the unloaded size of layers
   // that are required to execute this model
   size_t remained_size;
};

}
