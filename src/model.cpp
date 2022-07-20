#include <model.h>
#include <engine.h>
#include <util.h>
#include <deepcache.pb.h>

#include <torch/script.h>

static std::vector<ScriptModule> travel_layers(ScriptModule module, std::string name="") {
  std::vector<ScriptModule> traveled_layers;

  if (module.children().size() == 0) {
    traveled_layers.push_back(module);
    return traveled_layers;
  }
  else {
    for (auto name_child : module.named_children()) {
      if (name_child.name.find("drop") != std::string::npos) continue;
      if (name_child.name.find("relu") != std::string::npos) continue;
      auto layers = travel_layers(name_child.value, name_child.name);
      traveled_layers.insert(traveled_layers.end(), layers.begin(), layers.end());
    }
    return traveled_layers;
  }
}

Model::Model(const std::string name, const EngineType type, const std::vector<int> devices)
  : model_name(name),
    engine_type(type),
    devices(devices) {init();}

void Model::init() {
  const char *model_repo = getenv("MODEL_REPO");
  assert(model_repo);

  std::string model_prefix;
  std::string script_path;
  std::string config_path;

  model_prefix = std::string(model_repo) + "/" + model_name;
  script_path = model_prefix + "/model.pt";
  config_path = model_prefix + "/config.pbtxt";

  try {
    this->model = torch::jit::load(script_path);
    if (!util::read_from_pbtxt(this->model_config, config_path)) {
      std::stringstream msg;
      msg << "Failed to read " << config_path;
      throw std::runtime_error(msg.str());
    }
    std::cout << "Success model load\n";
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    throw e;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    throw e;
  }

  for (auto io : this->model_config.inputs())
    this->inputs.push_back(InputConfig(io));

  this->layers = travel_layers(this->model);
  this->model.eval();
  this->model.to(at::kCPU);
  this->model.cuda_host();

  switch (engine_type) {
    case EngineType::IN_MEMORY:
    case EngineType::ON_DEMAND:
    case EngineType::PIPESWITCH:
      this->n_layers = this->layers.size();
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
    auto split_iter = load_layer_idxs.begin();

    for (int i = 0; i < n_device; i++) {
      int device = devices[i];
      size_t cumm_size = 0;
      size_t layer_size = 0;
      std::vector<int> layer_list;

      for (auto iter = split_iter; iter != load_layer_idxs.end(); iter++) {
        layer_size = util::getModuleSize(layers[*iter]);
        cumm_size += layer_size;
        if (cumm_size > block_size) {
          split_iter = iter;
          break;
        }

        layer_list.push_back(*iter);
      }

      // Insert remain layers to last device
      if (i == n_device-1) {
        for (auto iter = split_iter; iter != load_layer_idxs.end(); iter++) {
          layer_list.push_back(*iter);
        }
      }

      device_map[device] = layer_list;
    }
  }

  // TODO
  // If using parallel transfer, the devices other than the target device
  // convert cuda_host to pin_memory

  this->model.cuda_backup();
  this->is_cuda = false;
}

void Model::forward(ScriptModuleInput& x) {
  engine::run(this, x);
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
  model.clear();
  is_cuda = false;
}
