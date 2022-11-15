#include <c10/cuda/CUDACachingAllocator.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <deepcache/model.h>

void ModelManager::add_model(std::string model_name, std::vector<int> devices) {
  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }
  std::string model_path = std::string(model_repo) + "/" + model_name;
  libtorch::Model *model;

  // FIXME: In DeepCache, the type of device is only one value, not array
  if (engine_type == EngineType::DEEPCACHE) {
    model = new deepcache::Model(model_name, model_path, devices[0]);
  }
  else {
    model = new deepplan::Model(model_name, model_path, engine_type, devices);
  }

  models.push_back(std::move(model));
}

void ModelManager::clear() {
  for (auto model : models) {
    model->clear();
  }
}

size_t getDeviceActiveMemorySize(int device){
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::DeviceStats;

  const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);

  return stats.active_bytes[0].current;
}
