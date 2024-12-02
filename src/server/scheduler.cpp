#include <server/scheduler.h>
#include <time_util.h>
#include <mutex>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

Scheduler::Scheduler(int device, const ServerOptions& options,
                     ModelPool* model_pool, std::string scheduler_name)
  : worker_(new Worker(device, options)),
    model_pool(model_pool),
    device(at::kCUDA, device),
    name(scheduler_name),
    options_(options) {
      if (name.empty()) {
        name = "Scheduler" + std::to_string(device);
      }
      running_models = new util::LRUCache<int, deepplan::Model*>();
      partial_models_list = std::list<util::LRUCache<int, deepplan::Model*>*>(10);
      for (auto& models_list : partial_models_list) {
        models_list = new util::LRUCache<int, deepplan::Model*>();
      }
      req_scoreboard = new RequestScoreboard(MAX_WINDOW_SIZE);
      cfr = new CFR();

    }


void Scheduler::enqueue_request(
    serverapi::InferenceRequest* request,
    std::function<void(serverapi::InferenceResponse*)> cb,
    std::function<void(serverapi::TimeoutResponse*)> timeout_cb) {
  int slo_ms = model_pool->get_model(request->model_id)->model_config.slo();
  request->deadline = request->arrival_time + slo_ms * 1e6;
  req_scoreboard->update_window(request->model_id);

  requests_.emplace(request, cb, timeout_cb);
}

void Scheduler::handle_requests() {
  {
    std::lock_guard<std::mutex> guard(tracker_mutex);
    std::shared_ptr<Completion> comp;

    for (; !completion_queue_.empty(); completion_queue_.pop()) {
      comp = completion_queue_.front();
      if (auto i_comp = std::dynamic_pointer_cast<InferCompletion>(comp)) {
        int action_id = i_comp->action_id;
        uint64_t end_time = i_comp->end_time;
        int model_id = i_comp->model_id;

        exec.update(action_id, end_time);

        auto model = model_pool->get_model(model_id);

        if ((--model_ref_cnts[model_id]) == 0) {
          running_models->put(model_id, model);
        }
      }
      else if (auto l_comp = std::dynamic_pointer_cast<LoadCompletion>(comp)) {
        int action_id = l_comp->action_id;
        uint64_t end_time = l_comp->end_time;
        uint64_t end_size = l_comp->end_size;
        mem.update(action_id, end_time, end_size);
      }
      else if (auto r_comp = std::dynamic_pointer_cast<ReclaimCompletion>(comp)) {
        int action_id = r_comp->action_id;
        uint64_t end_time = r_comp->end_time;
        uint64_t end_size = r_comp->end_size;

        mem.update(action_id, end_time, end_size);
      }
    }
  }

  auto now = util::now();
  if (!requests_.empty()) {
    handle_load(now);
    handle_exec(now);
  }
}

void Scheduler::handle_load(const uint64_t now) {
  InferTask task;

  if (!allow_prefetch) {
    uint64_t exec_at;
    exec_at = exec.available(now);

    uint64_t schedule_until = now + schedule_ahead;
    if (exec_at >= schedule_until) {
      return;
    }
  }

  uint64_t load_at;
  load_at = mem.available(now);

  uint64_t load_until = now + load_ahead;
  if (load_at >= load_until) {
    return;
  }

  task = *requests_.begin();
  int model_id = task.request->model_id;
  auto model = find_model(model_id);

  if (model == nullptr) {
    std::stringstream ss;
    ss << "Not found the model with id " << model_id << "\n";

    throw std::runtime_error(ss.str());
  }

  size_t uncached_size = model->uncached_size;
  if (uncached_size > 0) {
    uint64_t estimated_load_time = model->get_load_time() + lag;
    auto [loaded_size, load_layers] = model->load_layers();

    reclaim_gpu_memory(uncached_size);

    auto l_cb = [&mutex = tracker_mutex, &queue = completion_queue_](
        int action_id, uint64_t end_time, uint64_t end_size) {
      std::lock_guard<std::mutex> guard(mutex);
      queue.push(std::make_shared<LoadCompletion>(action_id, end_time, end_size));
    };
    mem.load_mem(action_seed_id, model_id, estimated_load_time, loaded_size);
    LoadAction action(action_seed_id, model_id, load_layers, l_cb);
    worker_->load(action);
    cold_model_ids.insert(model_id);

    action_seed_id++;
  }
}

void Scheduler::handle_exec(const uint64_t now) {
  InferTask task;

  uint64_t exec_at;

  exec_at = exec.available(now);

  uint64_t schedule_until = now + schedule_ahead;
  if (exec_at >= schedule_until) {
    return;
  }

  if (!allow_prefetch) {
    auto head_task = *requests_.begin();
    int model_id = head_task.request->model_id;
    auto model = model_pool->get_model(model_id);
    auto engine_type = model->engine_type;
    if (engine_type >= EngineType::PIPESWITCH) {
      if (model->uncached_size > 0) {
        return;
      }
      task = head_task;
    }
    else {
      // Skip scheduling the models that are not loaded.
      if (model->uncached_size > 0 || exec_at < mem.end_time(model_id)) {
        return;
      }
      task = head_task;
    }
  }
  else {
    for (auto it = requests_.begin(); it != requests_.end(); it++) {
      int model_id = it->request->model_id;
      auto model = model_pool->get_model(model_id);
      auto engine_type = model->engine_type;
      if (engine_type >= EngineType::PIPESWITCH) {
        auto batch_window = window_bufs[model_id].get_buf();
        assert(!batch_window.empty());

        auto min_batch_size = (*std::min_element(batch_window.begin(),
                                                 batch_window.end()));
        auto exec_time = model_pool->get_model_exec_time(model_id,
                                                         min_batch_size);
        auto end_load_time = mem.end_time(model_id);

        // If the pipeline can not hide the model, we schedule other models.
        if (model->uncached_size > 0 || (exec_at + exec_time) < end_load_time) {
          continue;
        }

        task = *it;
        break;
      }
      else {
        // Skip scheduling the models that are not loaded.
        if (model->uncached_size > 0 || exec_at < mem.end_time(model_id)) {
          continue;
        }

        task = *it;
        break;
      }
    }
  }

  if (!task.request) {
    return;
  }

  std::vector<InferTask> tasks;
  int model_id = task.request->model_id;
  auto model = find_model(model_id);
  if (model == nullptr) {
    std::stringstream ss;
    ss << "Not found the model with id " << model_id << "\n";

    throw std::runtime_error(ss.str());
  }

  int batch_size = 0;
  uint64_t end_load_time = mem.end_time(model_id);
  uint64_t deadline = task.request->deadline;

  int max_batch_size = model_pool->get_model(model_id)->model_config.max_batch_size();
  while (!requests_.empty() && batch_size < max_batch_size) {
    uint64_t next_estimated_time = model_pool->get_model_exec_time(model_id, batch_size+1) + lag;
    bool found = false;
    // If the SLO is violated, stop increasing the batch size.
    uint64_t complete_time = std::max(exec_at + next_estimated_time,
                                      end_load_time);
    if (deadline > complete_time ||
        (task.request->disable_timeout && tasks.empty())) {
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
    uint64_t exec_time = model->get_model_exec_time(batch_size) + lag;
    uint64_t estimated_time;
    if (end_load_time > exec_at + exec_time) {
      estimated_time = end_load_time - exec_at;
    }
    else {
      estimated_time = exec_time;
    }

    auto i_cb = [&mutex = tracker_mutex, &queue = completion_queue_](
        int action_id, uint64_t end_time, int model_id) {
      std::lock_guard<std::mutex> guard(mutex);
      queue.push(std::make_shared<InferCompletion>(action_id, end_time, model_id));
    };

    bool is_cold = false;
    if (auto search = cold_model_ids.find(model_id); search != cold_model_ids.end()) {
        cold_model_ids.erase(search);
        is_cold = true;
    }
    InferAction action(action_seed_id, model_id, is_cold, tasks, i_cb);

    exec.add_work(action_seed_id, estimated_time);

    action_seed_id++;

    model_ref_cnts[model_id]++;
    // Track the batch size for dynamic adjustment to the optimal point used
    // in reclaiming memory
    window_bufs[model_id].update(batch_size);

    worker_->infer(action);
  }
  else {
    // Requests violoting the deadline are handled in
    // handle_timout().
    timeouts_.push(task);
    requests_.erase(requests_.begin());
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

void Scheduler::reclaim_gpu_memory(const size_t size) {
  int64_t free_size = capacity_ - mem.get_mem();
  std::vector<ReclaimingOutput> outputs;
  int64_t required_mem  = static_cast<int64_t>(size);

  while (free_size < required_mem) {
    uint64_t mem_size = required_mem - free_size;
    auto output = preempt_model(mem_size);

    free_size += output.size;

    outputs.push_back(std::move(output));
  }

  if (!outputs.empty()) {
    auto r_cb = [&mutex = tracker_mutex, &queue = completion_queue_](
        int action_id, uint64_t end_time, uint64_t end_size) {
      std::lock_guard<std::mutex> guard(mutex);
      queue.push(
        std::make_shared<ReclaimCompletion>(action_id, end_time, end_size));
    };
    uint64_t reclaim_size = 0;
    for (auto output : outputs) {
      reclaim_size += output.size;
    }

    mem.reclaim_mem(action_seed_id, -1, reclaim_size);

    ReclaimAction action(action_seed_id, outputs, r_cb);
    worker_->reclaim(action);

    action_seed_id++;
  }
}

deepplan::Model* Scheduler::find_model(int model_id) {
  deepplan::Model* model = nullptr;

  if (running_models->exist(model_id)) {
    model = running_models->erase(model_id);
  }
  else {
    bool found = false;
    model = model_pool->get_model(model_id);

    if (r_policy_ == ReclaimPolicy::BALANCE) {
      // Balance or Hybrid reclaim policy should check if this model is in
      // partial models list. If the model is found, it should be cleared from that list
      auto iter = partial_models_list.begin();
      for (iter; iter != partial_models_list.end(); iter++) {
        if ((*iter)->exist(model_id)) {
          (*iter)->erase(model_id);
          found = true;
          break;
        }
      }
    }
    else if (r_policy_ == ReclaimPolicy::DYNAMIC ||
             r_policy_ == ReclaimPolicy::HYBRID) {
      auto second_models = partial_models_list.front();
      if (second_models->exist(model_id)) {
        second_models->erase(model_id);
      }
    }
  }

  return model;
}

ReclaimingOutput Scheduler::preempt_model(uint64_t mem_size) {
  ReclaimingOutput output;

  switch (r_policy_) {
    case ReclaimPolicy::RR:
      {
        if (running_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(running_models->pop(&evict_id));

          output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);

          auto uncached_size = evict_model->uncached_size;
          if ((evict_model->model_size - uncached_size) > 0) {
            running_models->put(evict_id, evict_model);
          }
        }
        else {
          throw std::runtime_error("There is no model to evict");
        }
      }

      break;
    case ReclaimPolicy::BALANCE:
      {
        auto models_list = partial_models_list;
        models_list.push_front(running_models);

        bool found = false;

        for (auto iter = models_list.begin(); iter != models_list.end(); iter++) {
          if ((*iter)->size() > 0) {
            int evict_id;
            auto evict_model = dynamic_cast<deepplan::Model*>((*iter)->pop(&evict_id));
            output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);
            found = true;

            iter++;
            if (iter != models_list.end()) {
              (*iter)->put(evict_id, evict_model);
            }
            else {
              // If the caching memory of the evict_model leaves on GPU,
              // we expand partial_models_list
              if (evict_model->uncached_size < evict_model->model_size) {
                auto partial_models = new util::LRUCache<int, deepplan::Model*>();
                partial_models->put(evict_id, evict_model);
                partial_models_list.push_back(std::move(partial_models));
              }
            }

            break;
          }
        }

        if (!found) {
          throw std::runtime_error("There is no model to evict");
        }
      }

      break;
    case ReclaimPolicy::HYBRID:
      {
        int evict_id;
        auto second_models = partial_models_list.front();

        // Apply LRU policy for models that don't reach the sweet spot.
        if (running_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(
              running_models->pop(&evict_id));

          auto batch_window = window_bufs[evict_id].get_buf();
          assert(!batch_window.empty());

          auto min_batch_size = (*std::min_element(batch_window.begin(),
                                                   batch_window.end()));

          size_t optimal_size = evict_model->optimal_sizes[min_batch_size - 1];

          uint64_t cached_rm_size = optimal_size - evict_model->uncached_size;
          output = model_pool->reclaim_model(
              evict_id, std::min(mem_size, cached_rm_size));

          if (evict_model->uncached_size < optimal_size) {
            running_models->put_back(evict_id, evict_model);
          }
          else {
            second_models->put(evict_id, evict_model);
          }

        }
        else if (second_models->size() > 0) {
          auto evict_model = dynamic_cast<deepplan::Model*>(
              second_models->pop(&evict_id));

          output = model_pool->reclaim_model(evict_id);
        }
        else {
          throw std::runtime_error("There is no model to evict");
        }
      }
      break;

    case ReclaimPolicy::DYNAMIC:
      {
        int evict_id;
        auto second_models = partial_models_list.front();

        // Apply LRU policy for models that don't reach the sweet spot.
        if (running_models->size() > 0) {
          auto evict_model = dynamic_cast<deepplan::Model*>(
              running_models->pop(&evict_id));

          auto batch_window = window_bufs[evict_id].get_buf();
          assert(!batch_window.empty());

          auto min_batch_size = (*std::min_element(batch_window.begin(),
                                                   batch_window.end()));

          size_t optimal_size = evict_model->optimal_sizes[min_batch_size - 1];

          uint64_t cached_rm_size = optimal_size - evict_model->uncached_size;
          output = model_pool->reclaim_model(
              evict_id, std::min(mem_size, cached_rm_size));

          if (evict_model->uncached_size < optimal_size) {
            running_models->put_back(evict_id, evict_model);
          }
          else {
            second_models->put(evict_id, evict_model);
          }
        }
        else if (second_models->size() > 0) {
          auto evict_model = dynamic_cast<deepplan::Model*>(
              second_models->pop(&evict_id));

          output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);

          auto uncached_size = evict_model->uncached_size;
          if (uncached_size < evict_model->model_size) {
            second_models->put(evict_id, evict_model);
          }
//          output = model_pool->reclaim_model(evict_id, mem_size);
//          if (evict_model->uncached_size < evict_model->model_size) {
//            second_models->put_back(evict_id, evict_model);
//          }
        }
        else {
          throw std::runtime_error("There is no model to evict");
        }
      }
      break;

    case ReclaimPolicy::LRU:
      {
        // Otherwise, LRU Eviction Policy
        int evict_id;
        auto evict_model = running_models->pop(&evict_id);
        output = model_pool->reclaim_model(evict_id);
      }
      break;

    default:
      break;
  }


  return output;
}

void Scheduler::clear_models() {
  while (running_models->size() > 0) {
    running_models->pop()->clear();
  }
  for (auto p_models : partial_models_list) {
    while (p_models->size() > 0) {
      p_models->pop()->clear();
    }
  }
  while (cfr->size() > 0) {
    auto [vruntime, model_id] = cfr->pop();
    model_pool->get_model(model_id)->clear();
  }
  req_scoreboard->clear();
  model_ref_cnts.clear();
  mem.clear();
}

void Scheduler::set_r_policy(ReclaimPolicy r_policy) {
  if (r_policy_ != r_policy) {
    r_policy_ = r_policy;
  }
}

void Scheduler::sync_setup() {
  size_t free;
  size_t total;
  cudaSetDevice(device.index());

  c10::cuda::CUDACachingAllocator::emptyCache();
  cudaError_t err = cudaMemGetInfo(&free, &total);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo Error\n");
  }
  capacity_ = size_t(free * options_.watermark);
  std::cout << "Available GPU-" << int(device.index()) << " "
            << "memory: " << capacity_ / 1024 / 1024 / 1024 << " GB\n";

  c10::cuda::CUDACachingAllocator::emptyCache();

  int num_models = model_pool->get_num_models();
  req_scoreboard->resize(num_models);
  model_ref_cnts.resize(num_models, 0);
  window_bufs.resize(num_models);
  for (auto& buf : window_bufs) {
    buf.resize(10); // default size of window tracking the batch size is 10
    // Clear dumpy data
    for (int i = 0; i < 10; i++) {
      buf.update(1);
    }
  }

  std::vector<ModelInstance*> model_instances;
  for (auto model : model_pool->models) {
    model_instances.push_back(model->model_instance);
  }
  worker_->sync_setup(std::move(model_instances));
}

void Scheduler::stop() {
  worker_->stop();
  delete worker_;
}
