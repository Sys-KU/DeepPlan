#pragma once

#include <torch/script.h>
#include <vector>
#include <map>
#include <chrono>
#include <cstdint>
#include <deepplan.pb.h>
#include <google/protobuf/text_format.h>
#include <exception>

typedef std::vector<torch::jit::IValue> ScriptModuleInput;
typedef torch::jit::script::Module ScriptModule;

#define STEP_SIZE (1024*1024)
#define ALIGNMENT 8
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & ~(ALIGNMENT-1))

typedef enum
{
  IN_MEMORY = 0,
  ON_DEMAND,
  PIPESWITCH,
  DEEPPLAN,
  DEEPCACHE,
  NONE,
} EngineType;

struct InputConfig {
 public:
  InputConfig(ModelInput io)
    : shape(io.shape().begin(), io.shape().end()),
      data_type(io.data_type()) {};

  std::vector<int64_t> shape;
  DataType data_type;

  at::Tensor get(void* data, int batch_size) {
    auto shape_ = shape;
    shape_.insert(shape_.begin(), batch_size);

    auto options = torch::TensorOptions();
    switch (data_type) {
      case TYPE_FP32:
        options = options.dtype(torch::kFloat32);
        break;
      case TYPE_INT64:
        options = options.dtype(torch::kInt64);
        break;
    }
    return torch::from_blob(data, shape_, options);
  }
};

namespace util {

typedef std::chrono::steady_clock::time_point time_point;

time_point hrt();

std::uint64_t now();

std::uint64_t nanos(time_point t);

template <typename T>
bool read_from_pbtxt(T& config, const std::string path) {
  std::ifstream fin(path);
  if (!fin.is_open()) return false;
  std::stringstream ss;
  ss << fin.rdbuf();
  return google::protobuf::TextFormat::ParseFromString(ss.str(), &config);
}

size_t getModuleSize(ScriptModule module, bool ignore_cuda=false);

class InputGenerator {
 public:
  InputGenerator();

  InputGenerator(const char* model_repo);

  void generate_input(std::string model_name, int batch_size, ScriptModuleInput* out);

  void generate_input(std::string model_name, int batch_size, std::vector<char>* out);

 private:
  void generate_rdata(size_t size, DataType data_type, char** buf_ptr);

  void generate_rdata(size_t size, DataType data_type, char* buf);

  void extend_rdata(DataType data_type, size_t size);

  void add_input_config(const std::string& model_name);

  std::map<std::string, std::vector<InputConfig>> input_config_map;
  std::map<DataType, std::vector<char>> rdata_map;

  const char* model_repo_;
};

} // namespace util
