#include <server/scheduler.h>
#include <time_util.h>

Scheduler::Scheduler(int device, const ServerOptions& options,
                     ModelPool* model_pool, std::string scheduler_name)
  : worker_(new Worker(device, options, model_pool)),
    model_pool(model_pool),
    device(at::kCUDA, device),
    name(scheduler_name),
    options_(options) {
      if (name.empty()) {
        name = "Scheduler" + std::to_string(device);
      }
    }


void Scheduler::enqueue_request(
    serverapi::InferenceRequest* request,
    std::function<void(serverapi::InferenceResponse*)> cb,
    std::function<void(serverapi::TimeoutResponse*)> timeout_cb) {
  int slo_ms = model_pool->get_model(request->model_id)->model_config.slo();
  request->deadline = request->arrival_time + slo_ms * 1e6;
  requests_.emplace(request, cb, timeout_cb);
}

void Scheduler::handle_requests() {
  InferTask task;
  std::unordered_map<int, std::vector<InferTask>> task_maps;

  while (!requests_.empty()) {
    task = *requests_.begin();

    if (task.request->disable_timeout ||
        task.request->deadline >= util::now()) {
      int model_id = task.request->model_id;
      if (task_maps.find(model_id) == task_maps.end()) {
        task_maps[model_id] = {task};
      }
      else {
        task_maps[model_id].push_back(task);
      }
    }
    else {
      // Requests violoting the deadline are handled in
      // handle_timout().
      timeouts_.push(task);
    }
    requests_.erase(task);
  }
  for (auto& [model_id, tasks] : task_maps) {
    worker_->infer(tasks);
  }
}

void Scheduler::handle_timeouts() {
  InferTask timeout_task;
  while (!timeouts_.empty()) {
    timeout_task = timeouts_.front();

    auto response = new serverapi::TimeoutResponse();
    response->req_id = timeout_task.request->req_id;
    timeout_task.timeout_cb(response);

    timeouts_.pop();
  }
}

void Scheduler::clear_models() {
  worker_->clear_models();
}

void Scheduler::set_r_policy(ReclaimPolicy r_policy) {
  worker_->set_r_policy(r_policy);
}

void Scheduler::sync_setup() {
  worker_->sync_setup();
}

void Scheduler::stop() {
  worker_->stop();
  delete worker_;
}
