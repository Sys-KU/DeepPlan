#pragma once

#include <network/server_api.h>
#include <network/session.h>
#include <server/worker.h>
#include <util.h>

#include <thread>
#include <atomic>

class Controller {
 public:
  Controller(network::MessageQueue& messages);

  void init();

  void run();

  void shutdown();

  void setup_models(std::vector<std::string> model_name, int n_models, EngineType engine_type, int mp_size);

 private:
  std::atomic_bool alive;

  std::vector<Worker*> workers;

  network::MessageQueue& messages_;

  std::thread ctrl_thr;

  std::vector<std::string> model_names_;
  int n_models_ = 0;
  int mp_size_ = 0;
  EngineType engine_type_ = EngineType::NONE;
};
