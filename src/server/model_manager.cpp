#include <c10/cuda/CUDACachingAllocator.h>
#include <server/model_manager.h>
#include <deepplan/model.h>

void ModelManager::add_model(std::string model_name, std::vector<int> devices) {
  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }
  std::string model_path = std::string(model_repo) + "/" + model_name;

  deepplan::Model* model = new deepplan::Model(model_name, model_path, engine_type, devices);

  models.push_back(std::move(model));
}

deepplan::Model* ModelManager::get_model(int model_id) {
  return models[model_id];
}

void ModelManager::clear() {
  for (auto model : models) {
    model->clear();
    delete model;
  }
}

size_t getDeviceActiveMemorySize(int device){
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::DeviceStats;

  const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);

  return stats.active_bytes[0].current;
}
