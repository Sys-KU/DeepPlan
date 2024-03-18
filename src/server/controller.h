#pragma once

#include <network/server_api.h>
#include <network/session.h>
#include <server/worker.h>
#include <server/scheduler.h>
#include <util.h>
#include <options.h>

#include <thread>
#include <atomic>

class Controller {
 public:
  Controller(network::MessageQueue& messages, const ServerOptions& options);

  void init();

  void run();

  void shutdown();

  void setup_models(std::vector<std::string> model_name, int n_models,
                    EngineType engine_type, ReclaimPolicy r_policy,
                    int mp_size, int slo_ms);

 private:
  std::atomic_bool alive;

  std::vector<Scheduler*> schedulers;

  network::MessageQueue& messages_;

  std::thread ctrl_thr;

  std::vector<std::string> model_names_;
  int n_models_ = 0;
  int mp_size_ = 0;
  int slo_ms_ = 0;
  EngineType engine_type_ = EngineType::NONE;
  ReclaimPolicy r_policy_ = ReclaimPolicy::LRU;

  const ServerOptions& options_;
};
