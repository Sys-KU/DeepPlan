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
               std::string worker_name)
  : device(at::kCUDA, device),
    name(worker_name),
    options_(options),
    alive(true) {
      if (name.empty()) {
        name = "Worker" + std::to_string(device);
      }
      worker_thr = std::thread(std::bind(&Worker::run, this));
      loader_thr = std::thread(std::bind(&Worker::run_loader, this));

      util::bind_thread_to_numa_node(worker_thr, device);
      util::bind_thread_to_numa_node(loader_thr, device);
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
      double t1, t2;

      if (auto infer_action = std::dynamic_pointer_cast<InferAction>(action)) {
        int model_id = infer_action->model_id;

        auto model_instance = model_instances[model_id];

        ScriptModuleInput inputs;

        // FIXME(jinu)
        for (auto input_config : model_instance->input_configs) {
          std::vector<at::Tensor> input_tensors;
          for (const auto& task : infer_action->tasks) {
            input_tensors.push_back(
                input_config.get(task.request->input, task.request->batch_size).to(device));
          }
          inputs.push_back(torch::cat(input_tensors));
        }

        num_colds += infer_action->is_cold;

        t1 = util::now();
        model_instance->forward(inputs, {});

        t2 = util::now();

        infer_time = t2 - t1;

        infer_action->complete(infer_time);

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
    }
  }
}

void Worker::run_loader() {
  std::shared_ptr<Action> action;

  while (alive) {
    while (load_queue_.try_pop(action)) {
      if (auto load_action = std::dynamic_pointer_cast<LoadAction>(action)) {
        int model_id = load_action->model_id;

        auto model_instance = model_instances[model_id];

        model_instance->load_layers(load_action->layers);

        auto end_size = getDeviceActiveMemorySize(device.index());
        load_action->complete(end_size);
      }
      else if (auto reclaim_action = std::dynamic_pointer_cast<ReclaimAction>(action)) {
        auto outputs = reclaim_action->outputs;
        for (auto output : outputs) {
          model_instances[output.model_id]->reclaim_layers(output.layers);
        }
        auto end_size = getDeviceActiveMemorySize(device.index());
        reclaim_action->complete(end_size);
      }
    }
  }
}

void Worker::sync_setup(std::vector<ModelInstance*> model_instances) {
  this->model_instances = model_instances;
}

void Worker::stop() {
  alive = false;
  if (worker_thr.joinable())
    worker_thr.join();
  if (loader_thr.joinable())
    loader_thr.join();
}

void Worker::infer(InferAction infer_action) {
  queue_.push(std::make_shared<InferAction>(infer_action));
}

void Worker::reclaim(ReclaimAction reclaim_action) {
  load_queue_.push(std::make_shared<ReclaimAction>(reclaim_action));
}

void Worker::load(LoadAction load_action) {
  load_queue_.push(std::make_shared<LoadAction>(load_action));
}
