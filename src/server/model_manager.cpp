#include <c10/cuda/CUDACachingAllocator.h>
#include <server/model_manager.h>
#include <deepplan/model.h>

uint64_t ModelPool::get_model_exec_time(int model_id, int batch_size) {
  auto model = dynamic_cast<deepplan::Model*>(get_model(model_id));

  return model->get_model_exec_time(batch_size);
}

void ModelPool::add_model(std::string model_name, EngineType engine_type,
                          std::vector<int> devices) {
  std::string model_path = std::string(model_repo) + "/" + model_name;

  auto model = new deepplan::Model(model_name, model_path, engine_type, devices);

  models.push_back(std::move(model));
}

bool ModelManager::setup(std::vector<std::string> model_names, int num_models,
                         EngineType engine_type, int mp_size) {
  bool should_setup = false;

  // Update if the setting parameters are different
  if ((model_names_ != model_names) ||
      (num_models_ != num_models) ||
      (engine_type_ != engine_type) ||
      (mp_size_ != mp_size)) {
    should_setup = true;
  }

  if (should_setup) {
    int num_pools = model_pools.size();

    std::cout << "Setup total " << num_models << " new models with "
              << num_pools << " workers\n";

    std::stringstream msg;
    msg << "Models: ";
    for (int i = 0; i < model_names.size(); i++) {
      msg << model_names[i];
      if (i < (model_names.size() - 1)) {
        msg << ", ";
      }
      else {
        msg << "\n";
      }
    }
    std::cout << msg.str();

    std::vector<std::vector<int>> partitions(num_pools);
    for (int i = 0; i < num_pools; i++) {
      std::vector<int> p;
      for (int d = 0; d < mp_size; d++)
        p.push_back((i + 2*d) % num_pools);

      partitions[i] = p;
    }

    if ((model_names_ == model_names) &&
        (engine_type_ == engine_type) &&
        (mp_size_ == mp_size)) {
      int num_add_models = std::max(num_models - num_models_, 0);
      auto progressbar = util::progressbar(num_add_models, "Models setup");
      if (num_add_models > 0) {
        int num_models_per_pool = num_add_models / num_pools;
        for (int p = 0; p < model_pools.size(); p++) {
          for (int i = 0; i < num_models_per_pool; i++) {
            auto model_name = model_names[i % model_names.size()];
            model_pools[p]->add_model(model_name, engine_type, partitions[p]);
            progressbar.update();
          }
        }
      }
    }
    else {
      auto progressbar = util::progressbar(num_models, "Models setup");
      int num_models_per_pool = num_models / num_pools;
      for (int p = 0; p < model_pools.size(); p++) {
        model_pools[p]->reset_models();
        for (int i = 0; i < num_models_per_pool; i++) {
          auto model_name = model_names[i % model_names.size()];
          model_pools[p]->add_model(model_name, engine_type, partitions[p]);
          progressbar.update();
        }
      }
    }

    // Update the changed model information.
    model_names_ = model_names;
    num_models_ = num_models;
    engine_type_ = engine_type;
    mp_size_ = mp_size;
  }

  return should_setup;
}

size_t getDeviceActiveMemorySize(int device){
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::DeviceStats;

  const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);

  return stats.active_bytes[0].current;
}
