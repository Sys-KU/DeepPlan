#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>
#include <deepplan.pb.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/script.h>

namespace deepplan {

ModelInstance::ModelInstance(const std::string script_path,
                             const std::vector<InputConfig> input_configs,
                             const at::Device device,
                             const std::vector<int> load_layers)
  : input_configs(input_configs),
    target_device(device) {
    init(script_path, load_layers);
  }

void ModelInstance::init(const std::string script_path,
                         const std::vector<int> load_layers) {
  try {
    this->model = torch::jit::load(script_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
  }

  this->layers = util::travel_layers(this->model);
  this->model.eval();

  {
    c10::cuda::CUDAGuard device_guard(this->target_device);
    this->model.to(at::kCPU);
    this->model.cuda_host();
  }


  for (auto i : load_layers) {
    this->layers[i].to(at::kCPU);
    this->layers[i].pin_memory();
    this->layers[i].cuda_backup();

  }

  this->model.cuda_backup();
}

torch::jit::IValue ModelInstance::forward(
    ScriptModuleInput& x, const std::vector<int> load_layer_idxs) {
  std::unordered_map<int, std::vector<ScriptModule>> device_map;
  std::vector<ScriptModule> load_layers;

  // TODO(jinu): Supprot the multi-device mapping
  for (auto i : load_layer_idxs) {
    load_layers.push_back(this->layers[i]);
  }
  device_map[target_device.index()] = load_layers;

  auto outputs = RunEngine(model, x, target_device, device_map);
  return outputs;
}

void ModelInstance::reclaim_layers(std::vector<int> reclaiming_layers) {
  for (int i : reclaiming_layers) {
    layers[i].clear();
  }
}

void ModelInstance::load_layers(std::vector<int> load_layer_idxs) {
  std::unordered_map<int, std::vector<ScriptModule>> device_map;
  std::vector<ScriptModule> load_layers;

  // TODO(jinu): Supprot the multi-device mapping
  for (auto i : load_layer_idxs) {
    load_layers.push_back(this->layers[i]);
  }
  device_map[target_device.index()] = load_layers;

  LoadLayers(target_device, device_map);
}

void ModelInstance::to(at::Device device, bool non_blocking) {
  model.to(device, non_blocking);
}

Model::Model(const std::string name, const std::string model_path, const EngineType type, const std::vector<int> devices)
  : engine_type(type),
    libtorch::Model(name, model_path, devices[0]) {
      init();
    }

void Model::init() {
  Prof::EngineType proto_type;
  if (engine_type == EngineType::PIPESWITCH) {
    proto_type = Prof::PIPESWITCH;
  }
  else if (engine_type == EngineType::DEEPPLAN) {
    proto_type = Prof::DEEPPLAN;
  }

  for (auto prof_ : this->model_config.profs()) {
    if (proto_type == prof_.engine_type()) {
      this->prof = prof_;
    }
  }
  std::vector<double> layer_load_times;
  for (auto load_time : prof.layer_load_times()) {
    layer_load_times.push_back(load_time);
  }
  std::vector<double> layer_sizes;
  for (auto layer_size : prof.layer_sizes()) {
    layer_sizes.push_back(layer_size);
  }

  this->n_layers = layer_sizes.size();

  std::vector<int> load_layers;
  switch (engine_type) {
    case EngineType::IN_MEMORY:
    case EngineType::ON_DEMAND:
    case EngineType::PIPESWITCH:
      for (int i = 0; i < this->n_layers; i++) {
        load_layers.push_back(i);
        this->load_state_maps.emplace_back(
            i, Device::CPU, layer_sizes[i], layer_load_times[i]);
      }

      for (auto optimal_point : prof.optimal_points()) {
        this->optimal_sizes.push_back(optimal_point.load_size());
        this->optimal_idxs.push_back(optimal_point.layer_idx());
        break;
      }
      break;

    case EngineType::DEEPPLAN:
    case EngineType::DEEPCACHE:
      for (auto plan : this->model_config.plans()) {
        if (Plan::DYNAMIC == plan.plan_type()) {
          auto ll = plan.load_layers();
          for (auto i : ll) {
            load_layers.push_back(i);
            this->load_state_maps.emplace_back(
                i, Device::CPU, layer_sizes[i], layer_load_times[i]);
          }
          break;
        }
      }

      for (auto optimal_point : prof.optimal_points()) {
        this->optimal_sizes.push_back(optimal_point.load_size());
        this->optimal_idxs.push_back(optimal_point.layer_idx());
      }
      break;
    default:
      std::cerr << "Found incorrect EngineType\n";
      break;
  }

  // Set device_map
  size_t model_size = 0;
  for (auto load_state: load_state_maps) {
    model_size += load_state.size;
  }
  this->model_size = model_size;
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

  uncached_size = model_size;

  model_instance = new ModelInstance(script_path, input_configs, target_device, load_layers);
}

torch::jit::IValue Model::forward(ScriptModuleInput& x) {
  std::vector<int> load_layers;
  for (auto load_state : load_state_maps) {
    if (load_state.device == Device::CPU) {
      load_layers.push_back(load_state.idx);
    }
  }
  auto outputs = model_instance->forward(x, load_layers);
  return outputs;
}

void Model::to(at::Device device, bool non_blocking) {
  model_instance->to(device, non_blocking);
  Device dest_device = device.is_cuda() ? Device::CUDA : Device::CPU;

  for (auto& iter : load_state_maps) {
    iter.device = dest_device;
  }

}

void Model::clear()
{
  size_t reclaimed_size = 0;
  std::vector<int> reclaiming_layers;
  for (auto& load_state : load_state_maps) {
    if (load_state.device == Device::CUDA) {
      reclaiming_layers.push_back(load_state.idx);
      load_state.device = Device::CPU;
      reclaimed_size += load_state.size;
    }
  }
  uncached_size = model_size;

  model_instance->clear();
}

std::pair<size_t, std::vector<int>> Model::reclaim_memory(size_t size) {
  size_t reclaimed_size = 0;

  // Reclaim memory in the reverse order of layerslayers backward.
  std::vector<int> reclaiming_layers;
  for (auto iter = load_state_maps.rbegin(); iter != load_state_maps.rend(); iter++) {
    auto&& load_state = (*iter);
    if (load_state.device == Device::CUDA) {
      reclaiming_layers.push_back(load_state.idx);
      load_state.device = Device::CPU;
      reclaimed_size += load_state.size;
      if (reclaimed_size > size) {
        break;
      }
    }
  }

  uncached_size += reclaimed_size;

  return std::make_pair(reclaimed_size, reclaiming_layers);
}

void Model::reclaim_memory(double rate) {
  reclaim_memory((size_t)(model_size * rate));
}

std::pair<size_t, std::vector<int>> Model::load_layers(bool non_blocking) {
  return load_layers(load_state_maps.size(), non_blocking);
}

std::pair<size_t, std::vector<int>> Model::load_layers(int n_layers, bool non_blocking) {
  int cnt = 0;
  size_t loaded_size = 0;

  std::vector<int> load_layers;
  for (auto& load_state : load_state_maps) {
    if (n_layers <= cnt) break;
    if (load_state.device == Device::CPU) {
      load_layers.push_back(load_state.idx);
      load_state.device = Device::CUDA;
      loaded_size += load_state.size;
      cnt++;
    }
  }

  uncached_size -= loaded_size;

  return std::make_pair(loaded_size, load_layers);
}

uint64_t Model::get_load_time() {
  double load_time = 0.f;
  for (auto load_state : load_state_maps) {
    if (load_state.device == Device::CPU) {
      load_time += load_state.load_time;
    }
  }

  uint64_t load_time_ns = static_cast<uint64_t>(load_time);

  return load_time_ns;
}

uint64_t Model::get_model_exec_time(int batch_size) {
  uint64_t exec_ns;
  for (auto exec_time : prof.exec_times()) {
    if (exec_time.batch_size() == batch_size) {
      exec_ns = exec_time.exec_ns();
      break;
    }
  }

  return std::max(exec_ns, get_load_time());
}

}
