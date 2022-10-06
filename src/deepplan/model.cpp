#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>
#include <deepplan.pb.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/script.h>

namespace deepplan {

Model::Model(const std::string name, const std::string model_path, const EngineType type, const std::vector<int> devices)
  : engine_type(type),
    libtorch::Model(name, model_path, devices[0]) {
      init();
    }

void Model::init() {
  {
    c10::cuda::CUDAGuard device_guard(this->target_device);
    this->model.to(at::kCPU);
    this->model.cuda_host();
  }

  switch (engine_type) {
    case EngineType::IN_MEMORY:
    case EngineType::ON_DEMAND:
    case EngineType::PIPESWITCH:
      for (int i = 0; i < this->n_layers; i++) {
        this->load_layer_idxs.push_back(i);
      }
      break;

    case EngineType::DEEPPLAN:
      for (auto plan : this->model_config.plans()) {
        if (Plan::DYNAMIC == plan.plan_type()) {
          auto ll = plan.load_layers();
          this->load_layer_idxs = std::vector<int>(ll.begin(), ll.end());
          break;
        }
      }
      break;
    default:
      std::cerr << "Found incorrect EngineType\n";
      break;
  }

  for (auto& i : this->load_layer_idxs) {
    this->layers[i].to(at::kCPU);
    this->layers[i].pin_memory();
  }

  // Set device_map
  this->model_size = util::getModuleSize(this->model, true);
  {
    int n_device = devices.size();
    size_t block_size = model_size / n_device;
    auto iter = load_layer_idxs.begin();

    for (int i = 0; i < n_device; i++) {
      int device = devices[i];
      size_t cumm_size = 0;
      size_t layer_size = 0;
      std::vector<int> layer_list;

      for (iter; iter != load_layer_idxs.end(); iter++) {
        layer_size = util::getModuleSize(layers[*iter]);
        cumm_size += layer_size;
        if (cumm_size > block_size) {
          break;
        }

        layer_list.push_back(*iter);
      }

      // Insert remain layers to last device
      if (i == n_device-1) {
        for (iter; iter != load_layer_idxs.end(); iter++) {
          layer_list.push_back(*iter);
        }
      }

      device_map[device] = layer_list;
    }
  }

  // TODO
  // If using parallel transfer, the devices other than the target device
  // convert cuda_host to pin_memory

  model.cuda_backup();
  this->is_cuda = false;
}

torch::jit::IValue Model::forward(ScriptModuleInput& x) {
  auto outputs = RunEngine(this, x);
  return outputs;
}

void Model::to(at::Device device, bool non_blocking) {
  model.to(device, non_blocking);
  if (device.is_cuda())
    is_cuda = true;
  else
    is_cuda = false;
}

void Model::clear()
{
  if (this->is_cuda) {
    model.clear();
    is_cuda = false;
  }
}

}
