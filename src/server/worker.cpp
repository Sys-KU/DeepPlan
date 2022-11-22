#include <torch/cuda.h>
#include <util.h>
#include <server/worker.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <deepcache/model.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

Worker::Worker(int device)
  : device(at::kCUDA, device),
    alive(true) {
      worker_thr = std::thread(std::bind(&Worker::run, this));
      running_models = new util::LRUCache<int, libtorch::Model*>();
      partial_models = new util::LRUCache<int, libtorch::Model*>();
    }

void Worker::run() {
  torch::NoGradGuard no_grad;

  InferTask task;

  while (alive) {
    while (queue_.try_pop(task)) {
      auto request = task.request;
      auto response = new serverapi::InferenceResponse();
      bool is_cold = false;
      double t1, t2;

      int model_id = request->model_id;
      libtorch::Model* model;

      if (running_models->exist(model_id)) {
        model = running_models->get(model_id);
      }
      else {
        auto new_model = model_manager->get_model(request->model_id);
        if (auto new_dc_model = dynamic_cast<deepcache::Model*>(new_model)) {
          // DeepCache Eviction Policy
          if (partial_models->exist(model_id)) {
            partial_models->erase(model_id);
          }

          while ((getDeviceActiveMemorySize(device.index())+new_dc_model->remained_size)
                 >= capacity_) {

            // We control the number of full-cached models that we can keep in Device.
            if (running_models->size() > 0) {
              int evict_id;
              auto evict_model = dynamic_cast<deepcache::Model*>(running_models->pop(&evict_id));
              evict_model->reclaim_layers(40);
              partial_models->put(evict_id, evict_model);
            }
            else if (partial_models->size() > 0) {
              auto evict_model = dynamic_cast<deepcache::Model*>(partial_models->pop());
              evict_model->clear();
            }
            else {
              throw "There is no model to evict";
              break;
            }
          }
        }
        else {
          // DeepPlan Eviction Policy
          while ((getDeviceActiveMemorySize(device.index())+new_model->model_size)
                 >= capacity_) {
            auto evict_model = running_models->pop();
            evict_model->clear();
          }
        }

        is_cold = true;
        running_models->put(model_id, new_model);
        model = new_model;
      }

      ScriptModuleInput inputs;

      for (auto input_config : model->input_configs) {
        inputs.push_back(
            input_config.get(request->input, request->batch_size).to(device));
      }

      t1 = util::now();
      model->forward(inputs);

      torch::cuda::synchronize(device.index());
      t2 = util::now();

      response->req_id = request->req_id;
      response->is_cold = is_cold;
      response->infer_time = t2 - t1;
      task.cb(response);
    }
  }
}

void Worker::init_model_manager(EngineType engine_type) {
  if (model_manager == nullptr) {
    model_manager = new ModelManager(engine_type);
  }
}

void Worker::add_models(std::vector<std::string> model_names, int n_models,
                        EngineType engine_type, std::vector<int> devices) {
  if (model_manager) {
    for (int i = 0; i < n_models; i++) {
      auto model_name = model_names[i % model_names.size()];
      model_manager->add_model(model_name, devices);
    }
  }
  else {
    throw "model_manager should be initialized before add_modles()\n";
  }
  size_t free;
  size_t total;
  size_t padding_size = (size_t)(3.0 * (1 << 30)); // 2GB
  cudaError_t err = cudaMemGetInfo(&free, &total);
  if (err != cudaSuccess) {
    throw "cudaMemGetInfo Error\n";
  }
  capacity_ = free - padding_size;
  std::cout << "capcity: " << capacity_ / 1024 / 1024 / 1024 << "\n";
}

void Worker::free_models() {
  if (model_manager) {
    clear_models();
    delete model_manager;

    model_manager = nullptr;
  }
}

void Worker::clear_models() {
  if (model_manager) {
    model_manager->clear();
    while (running_models->size() > 0) {
      running_models->pop();
    }
  }
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void Worker::stop() {
  alive = false;
  if (worker_thr.joinable())
    worker_thr.join();

  free_models();
}

void Worker::infer(
    serverapi::InferenceRequest* request,
    std::function<void(serverapi::InferenceResponse*)> cb) {
  InferTask task(request, cb);
  queue_.push(task);
}
