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
        this->load_state_maps.emplace_back(
            i, Device::CPU, util::getModuleSize(this->layers[i]));
      }

      for (auto prof : this->model_config.profs()) {
        if (Prof::PIPESWITCH == prof.engine_type()) {
            for (auto optimal_point : prof.optimal_points()) {
                this->optimal_size = optimal_point.load_size();
                this->optimal_idx = optimal_point.layer_idx();
                break;
            }
            break;
        }
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
            this->load_state_maps.emplace_back(
                i, Device::CPU, util::getModuleSize(this->layers[i]));
          }
          break;
        }
      }

      for (auto prof : this->model_config.profs()) {
        if (Prof::DEEPPLAN == prof.engine_type()) {
            for (auto optimal_point : prof.optimal_points()) {
                this->optimal_size = optimal_point.load_size();
                this->optimal_idx = optimal_point.layer_idx();
                break;
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
        layer_size = iter->size;
        cumm_size += layer_size;
        if (cumm_size > block_size) {
          break;
        }

        layer_list.push_back(iter->idx);
      }

      // Insert remain layers to last device
      if (i == n_device-1) {
        for (iter; iter != load_state_maps.end(); iter++) {
          layer_list.push_back(iter->idx);
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
    iter.device = dest_device;
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
    for (auto& load_state : load_state_maps) {
      if (load_state.device == Device::CUDA) {
        load_state.device = Device::CPU;
      }
    }
    is_cuda = false;
    uncached_size = model_size;
  }
}

void Model::reclaim_layers(int n_layers) {
  size_t reclaimed_size = 0;
  int cnt = 0;

  for (auto& load_state : load_state_maps) {
    if (load_state.device == Device::CUDA) {
      layers[load_state.idx].clear();
      load_state.device = Device::CPU;
      reclaimed_size += load_state.size;
      cnt++;
    }
    if (n_layers <= cnt) break;
  }

  uncached_size += reclaimed_size;
}

void Model::reclaim_memory(size_t size) {
  size_t reclaimed_size = 0;

  // Reclaim memory in the reverse order of layerslayers backward.
  for (auto iter = load_state_maps.rbegin(); iter != load_state_maps.rend(); iter++) {
    auto&& load_state = (*iter);
    if (load_state.device == Device::CUDA) {
      layers[load_state.idx].clear();
      load_state.device = Device::CPU;
      reclaimed_size += load_state.size;
      if (reclaimed_size > size) {
        break;
      }
    }
  }

  uncached_size += reclaimed_size;
}

void Model::reclaim_memory(double rate) {
  reclaim_memory((size_t)(model_size * rate));
}

void Model::load_layers(bool non_blocking) {
  load_layers(load_state_maps.size(), non_blocking);
}

void Model::load_layers(int n_layers, bool non_blocking) {
  int cnt = 0;

  for (auto& load_state : load_state_maps) {
    if (n_layers <= cnt) break;
    if (load_state.device == Device::CPU) {
      layers[load_state.idx].to(target_device, non_blocking);
      load_state.device = Device::CUDA;
      uncached_size -= load_state.size;
      cnt++;
    }
  }
}

}
