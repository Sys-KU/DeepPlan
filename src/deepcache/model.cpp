#include <deepcache/model.h>
#include <deepcache/engine.h>
#include <util.h>
#include <deepplan.pb.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/script.h>

namespace deepcache {

Model::Model(const std::string name, const std::string model_path, const int device)
  : libtorch::Model(name, model_path, device) {
      init();
    };

void Model::init() {
  for (auto& layer : this->layers) {
    layer.pin_memory();
    layer.cuda_backup();
  }

  this->layers_load_info.resize(n_layers);
  for (auto &device : this->layers_load_info) {
    device = Device::CPU;
  }

  model.cuda_backup();
  this->is_cuda = false;
  this->remained_size = model_size;
}

torch::jit::IValue Model::forward(ScriptModuleInput& x) {
  return RunEngine(this, x);
}

void Model::to(at::Device device, bool non_blocking) {
  model.to(device, non_blocking);
  Device dest_device = device.is_cuda() ? Device::CUDA : Device::CPU;

  for (auto& device : layers_load_info) {
    device = dest_device;
  }

  if (device.is_cuda()) {
    is_cuda = true;
    remained_size = 0;
  }
  else {
    is_cuda = false;
    remained_size = model_size;
  }
}

void Model::clear()
{
  if (this->is_cuda) {
    model.clear();
    for (auto& device : layers_load_info) {
      device = Device::CPU;
    }
    is_cuda = false;
    remained_size = model_size;
  }
}

void Model::reclaim_layers(int n_layers) {
  int cnt = 0;

  for (int i = layers_load_info.size()-1; i >= 0; i--) {
    if (layers_load_info[i] == Device::CUDA) {
      layers[i].clear();
      remained_size -= util::getModuleSize(layers[i]);
      layers_load_info[i] = Device::CPU;
      cnt++;
    }
    if (n_layers <= cnt) break;
  }
}

void Model::load_layers(int n_layers, bool non_blocking) {
  int cnt = 0;

  for (int i = 0; i < layers_load_info.size(); i++) {
    if (n_layers <= cnt) break;
    if (layers_load_info[i] == Device::CPU) {
      layers[i].to(target_device, non_blocking);
      remained_size += util::getModuleSize(layers[i]);
      layers_load_info[i] = Device::CUDA;
      cnt++;
    }
  }

  this->is_cuda = true;
}

std::vector<int> Model::get_host_layers() {
  std::vector<int> host_layers;
  for (int i = 0; i < layers_load_info.size(); i++) {
    if (layers_load_info[i] == Device::CPU) {
      host_layers.push_back(i);
    }
  }

  return host_layers;
}

}
