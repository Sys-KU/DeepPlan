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
        this->layers[i].to(at::kCPU);
        this->layers[i].pin_memory();
        this->layers[i].cuda_backup();
        this->load_state_maps.push_back(std::make_pair(i, Device::CPU));
      }
      break;

    case EngineType::DEEPPLAN:
    case EngineType::DEEPCACHE:
      for (auto plan : this->model_config.plans()) {
        if (Plan::DYNAMIC == plan.plan_type()) {
          auto ll = plan.load_layers();
          for (auto i : ll) {
            this->layers[i].to(at::kCPU);
            this->layers[i].pin_memory();
            this->layers[i].cuda_backup();
            this->load_state_maps.push_back(std::make_pair(i, Device::CPU));
          }
          break;
        }
      }
      break;
    default:
      std::cerr << "Found incorrect EngineType\n";
      break;
  }

  // Set device_map
  this->model_size = util::getModuleSize(this->model, true);
  {
    int n_device = devices.size();
    size_t block_size = model_size / n_device;
    auto iter = load_state_maps.begin();

    for (int i = 0; i < n_device; i++) {
      int device = devices[i];
      size_t cumm_size = 0;
      size_t layer_size = 0;
      std::vector<int> layer_list;

      for (iter; iter != load_state_maps.end(); iter++) {
        layer_size = util::getModuleSize(layers[iter->first]);
        cumm_size += layer_size;
        if (cumm_size > block_size) {
          break;
        }

        layer_list.push_back(iter->first);
      }

      // Insert remain layers to last device
      if (i == n_device-1) {
        for (iter; iter != load_state_maps.end(); iter++) {
          layer_list.push_back(iter->first);
        }
      }

      device_map[device] = layer_list;
    }
  }

  // TODO
  // If using parallel transfer, the devices other than the target device
  // convert cuda_host to pin_memory

  uncached_size = model_size;
  model.cuda_backup();
  this->is_cuda = false;
}

torch::jit::IValue Model::forward(ScriptModuleInput& x) {
  auto outputs = RunEngine(this, x);
  return outputs;
}

void Model::to(at::Device device, bool non_blocking) {
  model.to(device, non_blocking);
  Device dest_device = device.is_cuda() ? Device::CUDA : Device::CPU;

  for (auto& iter : load_state_maps) {
    iter.second = dest_device;
  }

  if (device.is_cuda())
    is_cuda = true;
  else
    is_cuda = false;
}

void Model::clear()
{
  if (this->is_cuda) {
    model.clear();
    for (auto& [idx, device] : load_state_maps) {
      if (device == Device::CUDA) {
        device = Device::CPU;
      }
    }
    is_cuda = false;
    uncached_size = model_size;
  }
}

void Model::reclaim_layers(int n_layers) {
  int cnt = 0;

  for (auto& [idx, device] : load_state_maps) {
    if (device == Device::CUDA) {
      layers[idx].clear();
      device = Device::CPU;
      uncached_size += util::getModuleSize(layers[idx]);
      cnt++;
    }
    if (n_layers <= cnt) break;
  }
}

void Model::reclaim_memory(size_t size) {
  size_t current_size = 0;

  // Reclaim memory in the reverse order of layerslayers backward.
  for (auto iter = load_state_maps.rbegin(); iter != load_state_maps.rend(); iter++) {
    auto&& [idx, device] = (*iter);
    if (device == Device::CUDA) {
      layers[idx].clear();
      device = Device::CPU;
      current_size += util::getModuleSize(layers[idx]);
      if (current_size > size) {
        break;
      }
    }
  }

  uncached_size += current_size;
}

void Model::reclaim_memory(double rate) {
  reclaim_memory((size_t)(model_size * rate));
}

void Model::load_layers(bool non_blocking) {
  load_layers(load_state_maps.size(), non_blocking);
}

void Model::load_layers(int n_layers, bool non_blocking) {
  int cnt = 0;

  for (auto& [idx, device] : load_state_maps) {
    if (n_layers <= cnt) break;
    if (device == Device::CPU) {
      layers[idx].to(target_device, non_blocking);
      device = Device::CUDA;
      uncached_size -= util::getModuleSize(layers[idx]);
      cnt++;
    }
  }
}

}
