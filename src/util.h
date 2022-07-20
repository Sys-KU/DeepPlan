#pragma once

#include <torch/script.h>
#include <vector>
#include <chrono>
#include <cstdint>
#include <deepcache.pb.h>
#include <google/protobuf/text_format.h>
#include <exception>

typedef std::vector<torch::jit::IValue> ScriptModuleInput;
typedef torch::jit::script::Module ScriptModule;

typedef enum
{
  IN_MEMORY = 0,
  ON_DEMAND,
  PIPESWITCH,
  DEEPPLAN,
  DEEPCACHE,
} EngineType;

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

} // namespace util
