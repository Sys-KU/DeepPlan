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
      running_models = new util::LRUCache<int, libtorch::Model*>();
      partial_models_list = std::list<util::LRUCache<int, libtorch::Model*>*>(10);
      for (auto& models_list : partial_models_list) {
        models_list = new util::LRUCache<int, libtorch::Model*>();
      }
      req_scoreboard = new RequestScoreboard(MAX_WINDOW_SIZE);
      cfr = new CFR();
    }

Worker::~Worker() {
  delete running_models;
  for (auto& models_list : partial_models_list) {
    delete models_list;
  }
}

void Worker::run() {
  torch::NoGradGuard no_grad;

  InferAction infer_action;

  double infer_time = 0;
  double total_sched_time = 0;
  double total_infer_time = 0;
  int num_reqs = 0;
  int num_colds = 0;
  double last_logging_time = 0.f;

  while (alive) {
    while (queue_.try_pop(infer_action)) {
      auto response = new serverapi::InferenceResponse();
      bool is_cold = false;
      double t1, t2;

      int model_id = infer_action.model_id;

      auto model = find_model(model_id, &is_cold);
      if (model == nullptr) {
        std::stringstream ss;
        ss << "Not found the model with id " << model_id << "\n";

        throw std::runtime_error(ss.str());
      }

      req_scoreboard->update_window(model_id);

      t1 = util::now();

      size_t uncached_size = dynamic_cast<deepplan::Model*>(model)->uncached_size;
      while ((getDeviceActiveMemorySize(device.index()) + uncached_size)
             >= capacity_) {
        preempt_models();
      }

      t2 = util::now();
      total_sched_time += ((t2-t1) / 1e6);
      if (is_cold == true) {
        num_colds++;
      }

      ScriptModuleInput inputs;

      for (auto input_config : model->input_configs) {
        std::vector<at::Tensor> input_tensors;
        for (const auto& task : infer_action.tasks) {
          input_tensors.push_back(
              input_config.get(task.request->input, task.request->batch_size).to(device));
        }
        inputs.push_back(torch::cat(input_tensors));
      }

      t1 = util::now();
      model->forward(inputs);

      torch::cuda::synchronize(device.index());
      t2 = util::now();

      infer_time = t2 - t1;

      running_models->put(model_id, model);

      infer_action.complete(infer_time, is_cold);

      total_infer_time += (infer_time / 1e6);
      num_reqs++;

      auto now = util::now() / 1e6;
      if ((now - last_logging_time) > 5000) {
        std::cout << "[INFO] ";
        std::cout << "infer time: " << total_infer_time / num_reqs << ", ";
        std::cout << "scheduling time: " << total_sched_time / num_reqs << " ms, ";
        std::cout << "num cold starts: " << num_colds << "\n";

        total_infer_time = 0;
        total_sched_time = 0;
        num_colds = 0;
        num_reqs = 0;

        last_logging_time = now;
      }
    }
  }
}

void Worker::set_r_policy(ReclaimPolicy r_policy) {
  if (r_policy_ != r_policy) {
    r_policy_ = r_policy;
  }
}

libtorch::Model* Worker::find_model(int model_id, bool* is_cold) {
  libtorch::Model* model = nullptr;

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

void Worker::preempt_models() {
  switch (r_policy_) {
    case ReclaimPolicy::RR:
      {
        if (running_models->size() > 0) {
          int evict_id;
          auto evict_model = dynamic_cast<deepplan::Model*>(running_models->pop(&evict_id));

          evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);

          auto uncached_size = evict_model->uncached_size;
          if ((evict_model->model_size - uncached_size)
              > evict_model->model_size * MINIMUM_CACHE_MEMORY_RATE) {
            running_models->put(evict_id, evict_model);
          }
          else {
            evict_model->clear();
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
            evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);
            found = true;

            iter++;
            if (iter != models_list.end()) {
              (*iter)->put(evict_id, evict_model);
            }
            else {
              // If the caching memory of the evict_model leaves on GPU,
              // we expand partial_models_list
              if (evict_model->uncached_size < evict_model->model_size) {
                auto partial_models = new util::LRUCache<int, libtorch::Model*>();
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
          evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);

          if (evict_model->uncached_size >= evict_model->optimal_size) {
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

          evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);

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
          evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);

          if (evict_model->uncached_size >= evict_model->optimal_size) {
            // Delegate the model to CFR.
            cfr->put(evict_id, req_scoreboard->get(evict_id));
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
          evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);

          assert(evict_model->uncached_size <= evict_model->model_size);
          if (evict_model->uncached_size < evict_model->model_size) {
            cfr->put(evict_model_id, vruntime + req_scoreboard->get(evict_model_id));
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
        auto evict_model = running_models->pop();
        evict_model->clear();
      }
      break;

    default:
      break;
  }
}

void Worker::clear_models() {
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
}

void Worker::sync_setup() {
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

  req_scoreboard->expand(model_pool->get_num_models());
}

void Worker::stop() {
  alive = false;
  if (worker_thr.joinable())
    worker_thr.join();
}

void Worker::infer(InferAction infer_action) {
  queue_.push(infer_action);
}
