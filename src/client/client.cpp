#include <iostream>

#include <client/client.h>
#include <util.h>
#include <server_api.h>

Client::Client()
  : alive(true),
    network_thr(std::bind(&Client::run, this)) {};

void Client::infer_async(std::vector<char>& input, int model_id,
                         std::function<void(serverapi::Response* rsp)> onSuccess) {
  serverapi::InferenceRequest request;

  request.model_id = model_id;
  request.batch_size = 1;
  request.input_size = input.size();
  request.input = input.data();

  session->send_request_async(request, onSuccess);
}

Client::~Client() {
  if (alive)
    shutdown();
}

serverapi::UploadModelResponse* Client::upload_model(std::string model_name, int n_models, EngineType engine_type, int mp_size) {
  serverapi::UploadModelRequest request;

  request.model_name = model_name;
  request.n_models = n_models;
  request.engine_type = engine_type;
  request.mp_size = mp_size;

  auto onSuccess = [this](serverapi::Response* rsp) {
    std::cout << "Success Upload\n";
  };

  auto response = dynamic_cast<serverapi::UploadModelResponse*>
                      (session->send_request(request, onSuccess));

  return response;
}

void Client::close() {
  serverapi::CloseRequest request;

  request.dummy = 0;

  auto onSuccess = [this](serverapi::Response* rsp) {
    std::cout << "Success Close\n";
  };

  auto response = session->send_request(request, onSuccess);
}

void Client::connect(const std::string& srv_ip, const std::string& port) {
  try {
    session = new network::ClientSession(io_service_);

    session->connect(srv_ip, port);
  }
 catch (std::exception& e) { io_service_.stop();
    std::cerr << e.what() << "\n";
  }

  return;
}

void Client::run() {
  while (alive) {
    try {
      boost::asio::io_service::work work(io_service_);
      io_service_.run();
    } catch (std::exception& e) {
      alive.store(false);
      std::cerr << "Exception in network thread: " << e.what();
    } catch (const char* m) {
      alive.store(false);
      std::cerr << "Exception in network thread: " << m;
    }
  }
}

void Client::shutdown() {
  session->await_completion();

  this->close();

  alive.store(false);
  io_service_.stop();
  if (network_thr.joinable())
    network_thr.join();
}
