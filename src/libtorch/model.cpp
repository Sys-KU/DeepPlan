#include <libtorch/model.h>
#include <util.h>
#include <deepplan.pb.h>
#include <c10/cuda/CUDAGuard.h>
#include <deepplan.pb.h>

#include <torch/script.h>

namespace libtorch {

Model::Model(const std::string name, const std::string model_path, const int device)
  : model_name(name),
    target_device(at::Device(at::kCUDA, device)) {
      init(model_path);
    };

void Model::init(const std::string model_path) {
  std::string script_name;
  std::string script_path;
  std::string config_path;

  if (!util::exists_dir(model_path.c_str())) {
    std::stringstream msg;
    msg << "Not found model_path '" << model_path << "'";
    throw std::runtime_error(msg.str());
  }

  {
    std::ostringstream ss;
    ss << "model" << int(target_device.index()) << ".pt";
    script_name = ss.str();
  }

  // FIXME should make sure the file path exist.
  script_path = model_path + "/" + script_name;
  config_path = model_path + "/config.pbtxt";

  try {
    this->model = torch::jit::load(script_path);
    if (!util::read_from_pbtxt(this->model_config, config_path)) {
      std::stringstream msg;
      msg << "Failed to read " << config_path;
      throw std::runtime_error(msg.str());
    }
    for (auto io : model_config.inputs()) {
      this->input_configs.emplace_back(io);
    }
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    throw e;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    throw e;
  }

  this->layers = util::travel_layers(this->model);
  this->n_layers = this->layers.size();
  this->model.eval();
  for (auto& layer : this->layers)
    layer.to(at::kCPU);

  this->model_size = util::getModuleSize(this->model, true);

  model.cuda_backup();
  this->is_cuda = false;
}

torch::jit::IValue Model::forward(ScriptModuleInput& x) {
  return model.forward(x);
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
