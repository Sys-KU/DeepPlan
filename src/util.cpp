#include <util.h>

namespace util {

std::uint64_t now() {
  return nanos(hrt());
}

time_point hrt()
{
  return std::chrono::steady_clock::now();
}

time_point epoch = hrt();

uint64_t epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();


std::uint64_t nanos(time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t - epoch).count() + epoch_time;
}

size_t getModuleSize(ScriptModule module, bool ignore_cuda) {
  size_t size = 0;
  for (auto param : module.parameters()) {
    if (ignore_cuda && param.is_cuda()) continue;
    size += param.nbytes();
  }
  return size;
}


InputGenerator::InputGenerator(const char* model_repo)
  : model_repo_(model_repo) {assert(model_repo);}

InputGenerator::InputGenerator()
  : model_repo_(getenv("PLAN_REPO")) {assert(model_repo_);}


void InputGenerator::extend_rdata(DataType data_type, size_t size) {
  auto it = rdata_map.find(data_type);
  auto& data = it->second;
  char rdata[ALIGN(size)];

  switch (data_type) {
    case TYPE_FP32:
      {
        for (int i = 0; i < size; i += sizeof(float)) {
          float value = (float)rand() / RAND_MAX;
          memcpy(rdata+i, &value, sizeof(float));
        }
        break;
      }
    case TYPE_INT64:
      {
        for (int i = 0; i < size; i += sizeof(int64_t)) {
          int64_t value = (int64_t)rand() % 30522;
          memcpy(rdata+i, &value, sizeof(int64_t));
        }
        break;
      }
    default:
      throw std::runtime_error("Incorrect DataType");
      break;
  }

  data.insert(data.begin(), rdata, rdata+sizeof(rdata));
}

void InputGenerator::generate_rdata(size_t size, DataType data_type, char** buf_ptr) {
  *buf_ptr = new char[size];
  generate_rdata(size, data_type, *buf_ptr);
}

void InputGenerator::generate_rdata(size_t size, DataType data_type, char* buf) {
  auto it = rdata_map.find(data_type);
  std::vector<char> rdata;

  if (it == rdata_map.end()) {
    it = rdata_map.insert({data_type, {}}).first;
  }

  if (it->second.size() < size) {
    size_t extend_size = std::max((size_t)STEP_SIZE, size);
    extend_rdata(data_type, extend_size);
  }

  rdata = it->second;

  // TODO select random range
  std::memcpy(buf, rdata.data(), size);
}

void InputGenerator::add_input_config(const std::string& model_name) {
  std::vector<InputConfig> input_configs;
  ModelConfig model_config;
  std::string model_prefix;
  std::string config_path;

  model_prefix = std::string(model_repo_) + "/" + model_name;
  config_path = model_prefix + "/config.pbtxt";

  try {
    if (!util::read_from_pbtxt(model_config, config_path)) {
      std::stringstream msg;
      msg << "Failed to read " << config_path;
      throw std::runtime_error(msg.str());
    }
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    throw e;
  }

  for (auto io : model_config.inputs()) {
    input_configs.emplace_back(io);
  }

  input_config_map.insert(std::make_pair(model_name, input_configs));
}

void InputGenerator::generate_input(std::string model_name, int batch_size, ScriptModuleInput* out) {
  ScriptModuleInput inputs;
  std::vector<InputConfig> input_configs;

  auto it = input_config_map.find(model_name);
  if (it == input_config_map.end()) {
    add_input_config(model_name);
    it = input_config_map.find(model_name);
  }

  input_configs = it->second;

  for (auto input_config : input_configs) {
    auto shape = input_config.shape;
    shape.insert(shape.begin(), batch_size);

    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    char* data;

    auto options = torch::TensorOptions();
    switch (input_config.data_type) {
      case TYPE_FP32:
        size *= sizeof(float);
        options = options.dtype(torch::kFloat32);
        break;
      case TYPE_INT64:
        size *= sizeof(int64_t);
        options = options.dtype(torch::kInt64);
        break;
    }
    generate_rdata(size, input_config.data_type, &data);

    inputs.push_back(torch::from_blob(data, shape, options));
  }


  *out = inputs;
}

// FIXME: Maybe convert double ptr
void InputGenerator::generate_input(std::string model_name, int batch_size, std::vector<char>* out) {
  std::vector<char> inputs;
  std::vector<InputConfig> input_configs;

  auto it = input_config_map.find(model_name);
  if (it == input_config_map.end()) {
    add_input_config(model_name);
    it = input_config_map.find(model_name);
  }

  input_configs = it->second;

  for (auto input_config : input_configs) {
    auto shape = input_config.shape;
    shape.insert(shape.begin(), batch_size);

    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    char* data;

    switch (input_config.data_type) {
      case TYPE_FP32:
        size *= sizeof(float);
        break;
      case TYPE_INT64:
        size *= sizeof(int64_t);
        break;
    }

    generate_rdata(size, input_config.data_type, &data);
    inputs.insert(inputs.end(), data, data+size);
  }

  *out = inputs;
}

bool exists_dir(const char* path) {
  struct stat info;

  if (stat(path, &info) != 0)
    return false;
  else if (info.st_mode & S_IFDIR)
    return true;
  else
    return false;
}

std::vector<ScriptModule> travel_layers(ScriptModule module, std::string name) {
  std::vector<ScriptModule> traveled_layers;

  if (module.children().size() == 0) {
    traveled_layers.push_back(module);
    return traveled_layers;
  }
  else {
    for (auto name_child : module.named_children()) {
      if (name_child.name.find("drop") != std::string::npos) continue;
      auto layers = travel_layers(name_child.value, name_child.name);
      traveled_layers.insert(traveled_layers.end(), layers.begin(), layers.end());
    }
    return traveled_layers;
  }
}

}
