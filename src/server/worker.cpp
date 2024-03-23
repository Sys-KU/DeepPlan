#include <torch/cuda.h>
#include <util.h>
#include <time_util.h>
#include <options.h>
#include <server/worker.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

Worker::Worker(int device, const ServerOptions& options,
               ModelPool* model_pool, std::string worker_name)
  : device(at::kCUDA, device),
    name(worker_name),
    options_(options),
    model_pool(model_pool),
    alive(true) {
      if (name.empty()) {
        name = "Worker" + std::to_string(device);
      }
      worker_thr = std::thread(std::bind(&Worker::run, this));
    }

Worker::~Worker() {
}

void Worker::run() {
  torch::NoGradGuard no_grad;

  std::shared_ptr<Action> action;

  double infer_time = 0;
  double total_infer_time = 0;
  int num_reqs = 0;
  int num_colds = 0;
  double last_logging_time = 0.f;

  while (alive) {
    while (queue_.try_pop(action)) {
      auto response = new serverapi::InferenceResponse();
      bool is_cold = false;
      double t1, t2;

      int model_id = action->model_id;

      auto model = model_pool->get_model(model_id);

      if (auto infer_action = std::dynamic_pointer_cast<InferAction>(action)) {
        ScriptModuleInput inputs;

        for (auto input_config : model->input_configs) {
          std::vector<at::Tensor> input_tensors;
          for (const auto& task : infer_action->tasks) {
            input_tensors.push_back(
                input_config.get(task.request->input, task.request->batch_size).to(device));
          }
          inputs.push_back(torch::cat(input_tensors));
        }

        if (!infer_action->layers.empty()) {
          is_cold = true;
        }
        num_colds += is_cold;

        t1 = util::now();
        model->model_instance->forward(inputs, infer_action->layers);

        torch::cuda::synchronize(device.index());
        t2 = util::now();

        infer_time = t2 - t1;

        auto end_size = getDeviceActiveMemorySize(device.index());
        infer_action->complete(infer_time, end_size, is_cold);

        total_infer_time += (infer_time / 1e6);
        num_reqs++;

        auto now = util::now() / 1e6;
        if ((now - last_logging_time) > 5000) {
          std::cout << "[INFO] ";
          std::cout << "infer time: " << total_infer_time / num_reqs << ", ";
          std::cout << "num cold starts: " << num_colds << "\n";

          total_infer_time = 0;
          num_colds = 0;
          num_reqs = 0;

          last_logging_time = now;
        }
      }
      else if (auto reclaim_action = std::dynamic_pointer_cast<ReclaimAction>(action)) {
        model->model_instance->reclaim_layers(reclaim_action->layers);
        auto end_size = getDeviceActiveMemorySize(device.index());
        reclaim_action->complete(end_size);
      }
    }
  }
}

void Worker::sync_setup(std::vector<ModelInstance*> model_instances) {
  model_instances = model_instances;
}

void Worker::stop() {
  alive = false;
  if (worker_thr.joinable())
    worker_thr.join();
}

void Worker::infer(InferAction infer_action) {
  queue_.push(std::make_shared<InferAction>(infer_action));
}

void Worker::reclaim(ReclaimAction reclaim_action) {
  queue_.push(std::make_shared<ReclaimAction>(reclaim_action));
}
