#include <server/scheduler.h>

Scheduler::Scheduler(int device, const ServerOptions& options, std::string scheduler_name)
  : worker_(new Worker(device, options)),
    device(at::kCUDA, device),
    name(scheduler_name),
    options_(options) {
      if (name.empty()) {
        name = "Scheduler" + std::to_string(device);
      }
    }


void Scheduler::enqueue_request(
    serverapi::InferenceRequest* request,
    std::function<void(serverapi::InferenceResponse*)> cb) {
  requests_.emplace(request, cb);
}

void Scheduler::handle_requests() {
  InferTask task;
  std::unordered_map<int, std::vector<InferTask>> task_maps;

  while (!requests_.empty()) {
    task = *requests_.begin();
    int model_id = task.request->model_id;
    if (task_maps.find(model_id) == task_maps.end()) {
      task_maps[model_id] = {task};
    }
    else {
      task_maps[model_id].push_back(task);
    }
    requests_.erase(task);
  }
  for (auto& [model_id, tasks] : task_maps) {
    worker_->infer(tasks);
  }
}

void Scheduler::clear_models() {
  worker_->clear_models();
}

void Scheduler::add_models(std::vector<std::string> model_names, int n_models,
                           EngineType engine_type, int slo_ms,
                           std::vector<int> devices) {
  worker_->add_models(model_names, n_models, engine_type, devices);
}

void Scheduler::reset_models(std::vector<std::string> model_names, int n_models,
                             EngineType engine_type, int slo_ms,
                             std::vector<int> devices) {
  worker_->free_models();
  worker_->init_model_manager(engine_type);
  worker_->add_models(model_names, n_models, engine_type, devices);
}

void Scheduler::set_r_policy(ReclaimPolicy r_policy) {
  worker_->set_r_policy(r_policy);
}

void Scheduler::stop() {
  worker_->stop();
  delete worker_;
}
