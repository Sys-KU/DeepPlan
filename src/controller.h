#pragma once

#include <network/server_api.h>
#include <network/session.h>
#include <util.h>

#include <thread>

class Controller {
 public:
  Controller(network::MessageQueue& messages, EngineType, int mp_size);

  void run();

  int mp_size;
  EngineType engine_type;

 private:
  network::MessageQueue& messages_;
  std::thread ctrl_thr;
};
