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
  InferTask task;
  std::unordered_map<int, std::vector<InferTask>> task_maps;

  while (!requests_.empty()) {
    uint64_t exec_at;
    uint64_t now = util::now();

    exec_at = exec.available();

    uint64_t schedule_until = now + schedule_ahead;
    if (exec_at >= schedule_until) {
      break;
    }

    task = *requests_.begin();

    std::vector<InferTask> tasks;
    int model_id = task.request->model_id;
    int batch_size = 0;
    uint64_t deadline = task.request->deadline;

    // FIXME(jinu): Add the max batchsize limitation.
    int max_batch_size = model_pool->get_model(model_id)->model_config.max_batch_size();
    while (!requests_.empty() && batch_size < max_batch_size) {
      uint64_t next_estimated_time = model_pool->get_model_exec_time(model_id, batch_size+1) + lag;
      bool found = false;
      // If the SLO is violated, stop increasing the batch size.
      if (deadline > (exec_at + next_estimated_time) ||
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
      bool is_cold = false;
      auto model = find_model(model_id, &is_cold);

      if (model == nullptr) {
        std::stringstream ss;
        ss << "Not found the model with id " << model_id << "\n";

        throw std::runtime_error(ss.str());
      }

      size_t uncached_size = model->uncached_size;
      uint64_t mem_size = mem.get_mem();
      uint64_t reclaimed_size = 0;
      std::vector<ReclaimingOutput> outputs;
      while ((mem_size + uncached_size - reclaimed_size)
             >= capacity_) {
        auto output = preempt_models();

        reclaimed_size += output.size;

        outputs.push_back(std::move(output));
      }

      if (!outputs.empty()) {
        auto r_cb = [&mem = mem](int action_id, uint64_t end_size) {
          mem.update(action_id, end_size);
        };

        mem.reclaim_mem(action_seed_id, reclaimed_size);

        ReclaimAction action(action_seed_id, outputs, r_cb);
        worker_->reclaim(action);

        action_seed_id++;
      }

      running_models->put(model_id, model);

      uint64_t estimated_time = model_pool->get_model_exec_time(model_id, batch_size) + lag;

      auto [loaded_size, load_layers] = model->load_layers();
      mem.load_mem(action_seed_id, loaded_size);

      auto i_cb = [&exec = exec, &mem = mem](int action_id, uint64_t end_time, uint64_t end_size) {
        exec.update(action_id, end_time);
        mem.update(action_id, end_size);
      };

      InferAction action(action_seed_id, model_id, load_layers, tasks, i_cb);

      exec.add_work(action_seed_id, estimated_time);

      action_seed_id++;

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

deepplan::Model* Scheduler::find_model(int model_id, bool* is_cold) {
  deepplan::Model* model = nullptr;

  if (running_models->exist(model_id)) {
    model = running_models->erase(model_id);
  }
  else {
    bool found = false;
    model = model_pool->get_model(model_id);

    if (r_policy_ == ReclaimPolicy::BALANCE ||
        r_policy_ == ReclaimPolicy::HYBRID) {
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
    else if (r_policy_ == ReclaimPolicy::DYNAMIC && cfr->exist(model_id)) {
      cfr->erase(model_id);
      found = true;
    }

    if (!found) {
      *is_cold = true;
    }
  }

  return model;
}

ReclaimingOutput Scheduler::preempt_models() {
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
        auto partial_models = partial_models_list.front();
        bool found = false;

        // Apply LRU policy for models that don't reach the sweet spot.
        if (running_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(
              running_models->pop(&evict_id));
          output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);

          auto batch_window = window_bufs[evict_id].get_buf();
          assert(!batch_window.empty());

          auto min_batch_size = (*std::min_element(batch_window.begin(),
                                                   batch_window.end()));

          size_t optimal_size = evict_model->optimal_sizes[min_batch_size - 1];

          if (evict_model->uncached_size >= optimal_size) {
            // Delegate the model to RR.
            partial_models->put(evict_id, evict_model);
          }
          else {
            running_models->put_back(evict_id, evict_model);
          }

          found = true;
        }

        if (!found && partial_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(
              partial_models->pop(&evict_id));

          output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);

          if (evict_model->model_size > evict_model->uncached_size) {
            partial_models->put(evict_id, evict_model);
          }
          else {
            evict_model->clear();
          }
          found = true;
        }

        if (!found) {
          throw std::runtime_error("There is no model to evict");
        }
      }
      break;

    case ReclaimPolicy::DYNAMIC:
      {
        bool found = false;

        // Apply LRU policy for models that don't reach the sweet spot.
        if (running_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(
              running_models->pop(&evict_id));
          output = model_pool->reclaim_model(evict_id, RECLAIM_MEMORY_STEP);

          auto batch_window = window_bufs[evict_id].get_buf();
          assert(!batch_window.empty());

          auto min_batch_size = (*std::min_element(batch_window.begin(),
                                                   batch_window.end()));

          size_t optimal_size = evict_model->optimal_sizes[min_batch_size - 1];

          if (evict_model->uncached_size >= optimal_size) {
            // Delegate the model to CFR.
            cfr->put(evict_id, req_scoreboard->get_score(evict_id));
          }
          else {
            running_models->put_back(evict_id, evict_model);
          }

          found = true;
        }

        if (!found && cfr->size() > 0)  {
          auto [vruntime, evict_model_id] = cfr->pop();

          auto evict_model = dynamic_cast<deepplan::Model*>(
              model_pool->get_model(evict_model_id));
          output = model_pool->reclaim_model(evict_model_id, RECLAIM_MEMORY_STEP);

          assert(evict_model->uncached_size <= evict_model->model_size);
          if (evict_model->uncached_size < evict_model->model_size) {
            cfr->put(evict_model_id, vruntime + req_scoreboard->get_score(evict_model_id));
          }

          found = true;
        }


        if (!found) {
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
  window_bufs.resize(num_models);
  for (auto& buf : window_bufs) {
    buf.resize(10); // default size of window trakcing the batch size is 10
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
