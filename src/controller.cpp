#include <controller.h>
#include <network/session.h>
#include <util.h>

#include <thread>

Controller::Controller(network::MessageQueue& messages, EngineType engine_type, int mp_size)
  : messages_(messages),
    engine_type(engine_type),
    mp_size(mp_size),
    ctrl_thr(std::bind(&Controller::run, this)) {};

void Controller::run() {
  while (true) {
    network::Message message;
    messages_.pop(message);

    if (auto infer = dynamic_cast<serverapi::InferenceRequest*>(message.req)) {
      auto response = new serverapi::InferenceResponse();
      response->req_id = infer->req_id;
      std::cout << "process infer\n";

      message.srv_session->send_response(response);
    }
  }
}
