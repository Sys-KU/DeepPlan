#include <torch/cuda.h>
#include <server/worker.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <cuda_runtime_api.h>

Worker::Worker(int device)
  : device(at::kCUDA, device),
    alive(true) {
      worker_thr = std::thread(std::bind(&Worker::run, this));
    }

void Worker::run() {
  torch::NoGradGuard no_grad;

  InferTask task;

  while (alive) {
    while (queue_.try_pop(task)) {
      auto request = task.request;
      auto response = new serverapi::InferenceResponse();
      bool is_cold = false;

      int model_id = request->model_id;
      deepplan::Model* model;

      if (running_models->exist(model_id)) {
        model = running_models->get(model_id);
      }
      else {
        auto new_model = model_manager->get_model(request->model_id);

        while (getDeviceActiveMemorySize(device.index()) >= capacity_) {
          auto evict_model = running_models->pop();
          evict_model->clear();
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

      model->forward(inputs);

      torch::cuda::synchronize(device.index());

      response->req_id = request->req_id;
      response->is_cold = is_cold;
      task.cb(response);
    }
  }
}

void Worker::init_model(std::vector<std::string> model_names, int n_models,
                        EngineType engine_type, std::vector<int> devices) {
  if (model_manager == nullptr) {
    size_t free;
    size_t total;
    size_t padding_size = (size_t)(5.5 * (1 << 30)); // 6GB
    int n_models_per = n_models / model_names.size();

    model_manager = new ModelManager(engine_type);
    for (auto model_name : model_names)
      for (int i = 0; i < n_models_per; i++)
        model_manager->add_model(model_name, devices);

    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
      throw "cudaMemGetInfo Error\n";
    }

    capacity_ = total - padding_size;
    running_models = new LRUCache<int, deepplan::Model*>();
  }
}

void Worker::reset_model() {
  if (model_manager) {
    model_manager->clear();
    delete model_manager;
    model_manager = nullptr;
    delete running_models;
  }
}

void Worker::stop() {
  alive = false;
  if (worker_thr.joinable())
    worker_thr.join();

  reset_model();
}

void Worker::infer(
    serverapi::InferenceRequest* request,
    std::function<void(serverapi::InferenceResponse*)> cb) {
  InferTask task(request, cb);
  queue_.push(task);
}
