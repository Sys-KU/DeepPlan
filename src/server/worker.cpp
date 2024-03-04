#include <torch/cuda.h>
#include <util.h>
#include <options.h>
#include <server/worker.h>
#include <server/model_manager.h>
#include <deepplan/model.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

Worker::Worker(int device, const ServerOptions& options, std::string worker_name)
  : device(at::kCUDA, device),
    name(worker_name),
    options_(options),
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

  InferTask task;

  while (alive) {
    while (queue_.try_pop(task)) {
      auto request = task.request;
      auto response = new serverapi::InferenceResponse();
      bool is_cold = false;
      double t1, t2;

      int model_id = request->model_id;

      auto model = find_model(model_id, &is_cold);
      if (model == nullptr) {
        std::stringstream ss;
        ss << "Not found the model with id " << model_id << "\n";

        throw std::runtime_error(ss.str());
      }

      req_scoreboard->update_window(model_id);

      size_t uncached_size = dynamic_cast<deepplan::Model*>(model)->uncached_size;
      while ((getDeviceActiveMemorySize(device.index()) + uncached_size)
             >= capacity_) {
        preempt_models();
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

      running_models->put(model_id, model);

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

void Worker::set_r_policy(ReclaimPolicy r_policy) {
  if (r_policy_ != r_policy) {
    r_policy_ = r_policy;
  }
}

void Worker::add_models(std::vector<std::string> model_names, int n_models,
                        EngineType engine_type, std::vector<int> devices) {
  auto progressbar = util::progressbar(n_models, name);
  if (model_manager) {
    req_scoreboard->expand(n_models);
    for (int i = 0; i < n_models; i++) {
      auto model_name = model_names[i % model_names.size()];
      model_manager->add_model(model_name, devices);
      progressbar.update();
    }
  }
  else {
    throw std::runtime_error("model_manager should be initialized before add_modles()\n");
  }
  size_t free;
  size_t total;
  cudaSetDevice(device.index());

  c10::cuda::CUDACachingAllocator::emptyCache();
  cudaError_t err = cudaMemGetInfo(&free, &total);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo Error\n");
  }
  capacity_ = size_t(free * options_.watermark);
  std::cout << "Available GPU memory: " << capacity_ / 1024 / 1024 / 1024 << " GB\n";
}

libtorch::Model* Worker::find_model(int model_id, bool* is_cold) {
  libtorch::Model* model = nullptr;

  if (running_models->exist(model_id)) {
    model = running_models->erase(model_id);

    // If this model is DeepCache and partial,
    // it should secure memory to load the model
    if (auto dc_model = dynamic_cast<deepplan::Model*>(model)) {
      if (dc_model->uncached_size > 0) {
        *is_cold = true;
      }
    }
  }
  else {
    bool found = false;
    model = model_manager->get_model(model_id);
    if (auto dc_model = dynamic_cast<deepplan::Model*>(model) &&
        r_policy_ == ReclaimPolicy::BALANCE) {
      // Balance reclaim policy should check if this model is in partial models list
      // If the model is found, it should be cleared from that list
      auto iter = partial_models_list.begin();
      for (iter; iter != partial_models_list.end(); iter++) {
        if ((*iter)->exist(model_id)) {
          (*iter)->erase(model_id);
          found = true;
          break;
        }
      }
    }

    if (!found && cfr->exist(model_id)) {
      cfr->erase(model_id);
    }

    *is_cold = true;
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
        auto models_list = partial_models_list;
        models_list.push_front(running_models);

        bool found = false;

        for (auto iter = models_list.begin(); iter != models_list.end(); iter++) {
          if ((*iter)->size() > 0) {
            int evict_id;
            auto evict_model = dynamic_cast<deepplan::Model*>((*iter)->pop(&evict_id));
            evict_model->reclaim_memory(RECLAIM_MEMORY_RATE);
            found = true;

            // NOTE(jinu): If uncached memory doesn't exceed the optimal point,
            // We pass the evict model to the following list. The evict priority
            // of that model is lowered. The method of passing to the following
            // list means managing things in a balanced manner. Otherwise, We
            // put the current list instead of passing. The models on the list
            // are managed as round-robin.
            if (evict_model->uncached_size < 180 * MB) {
              iter++;
            }

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

    case ReclaimPolicy::DYNAMIC:
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

            // NOTE(jinu): If uncached memory doesn't exceed the optimal point,
            // We pass the evict model to the following list. The evict priority
            // of that model is lowered. The method of passing to the following
            // list means managing things in a balanced manner. Otherwise, We
            // put the current list instead of passing. The models on the list
            // are managed as round-robin.
            if (evict_model->uncached_size < 180 * MB) {
              iter++;
            }
            else {
              cfr->put(evict_id, req_scoreboard->get(evict_id));
              break;
            }

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

        if (!found && cfr->size() > 0)  {
          auto [vruntime, evict_model_id] = cfr->pop();

          auto evict_model = dynamic_cast<deepplan::Model*>(
              model_manager->get_model(evict_model_id));
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
    for (auto p_models : partial_models_list) {
      while (p_models->size() > 0) {
        p_models->pop();
      }
    }
    while (cfr->size() > 0) {
      cfr->pop();
    }
    req_scoreboard->clear();
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
