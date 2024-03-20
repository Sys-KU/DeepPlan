#include <server/scheduler.h>
#include <time_util.h>
#include <mutex>

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
    uint64_t exec_at;
    uint64_t now = util::now();
    {
      std::lock_guard<std::mutex> guard(exec_mutex);
      exec_at = exec.available();
    }

    uint64_t schedule_until = now + schedule_ahead;
    if (exec_at >= schedule_until) {
      break;
    }

    task = *requests_.begin();

    std::vector<InferTask> tasks;
    int model_id = task.request->model_id;
    int batch_size = 0;
    uint64_t deadline = task.request->deadline;

    while (!requests_.empty()) {
      uint64_t next_estimated_time = model_pool->get_model_exec_time(model_id, batch_size+1);
      bool found = false;
      if (deadline > (exec_at + next_estimated_time) || task.request->disable_timeout) {
        for (auto it = requests_.begin(); it != requests_.end(); it++) {
          if (it->request->model_id == model_id) {
            tasks.push_back(*it);
            batch_size++;
            requests_.erase(it);
            found = true;
            break;
          }
        }
      }
      if (!found) {
        break;
      }
    }

    if (!tasks.empty()) {
      auto cb = [&exec = exec, &exec_mutex = exec_mutex](int action_id, uint64_t end_time) {
        std::lock_guard<std::mutex> guard(exec_mutex);
        exec.update(action_id, end_time);
      };
      InferAction action(action_seed_id, model_id, tasks, cb);

      uint64_t estimated_time = model_pool->get_model_exec_time(model_id, batch_size);
      {
        std::lock_guard<std::mutex> guard(exec_mutex);
        exec.add_work(action_seed_id, estimated_time);
      }

      action_seed_id++;
      worker_->infer(action);
    }
    else {
      // Requests violoting the deadline are handled in
      // handle_timout().
      timeouts_.push(task);
      requests_.erase(requests_.begin());
    }
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
