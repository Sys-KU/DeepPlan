#include <iostream>

#include <client.h>
#include <util.h>
#include <server_api.h>

Client::Client()
  : alive(true),
    network_thr(std::bind(&Client::run, this)) {};

void Client::infer(std::vector<uint8_t>& input, int model_id) {
  uint64_t t_send = util::now();

  auto onSuccess = [this, t_send]() {
    uint64_t t_receive = util::now();
    std::cout << "rcv success\n";
  };

  infer(input, model_id, onSuccess);
}

Client::~Client() {
  shutdown();
}

void Client::infer(std::vector<uint8_t>& input, int model_id, std::function<void(void)> onSuccess) {
  serverapi::InferenceRequest request;

  request.model_id = model_id;
  request.batch_size = 1;
  request.input_size = input.size();
  request.input = input.data();

  session->send_request(request, onSuccess);
}

void Client::close() {
  serverapi::CloseRequest request;

  request.dummy = 0;

  auto onSuccess = [this]() {
    std::cout << "Success Close\n";
  };

  session->send_request(request, onSuccess);
}

void Client::sync_close() {
  this->close();

  session->await_completion();
}

void Client::connect(const std::string& srv_ip, const std::string& port) {
  try {
    session = new network::ClientSession(io_service_);

    session->connect(srv_ip, port);
  }
  catch (std::exception& e) {
    io_service_.stop();
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

  this->sync_close();

  alive.store(false);
  io_service_.stop();
  if (network_thr.joinable())
    network_thr.join();
}

int main(int argc, char** argv) {
  std::string model_repo = "/home/jinu/DeepCache/plans/A5000";

  try {
    Client client;

    client.connect("127.0.0.1", "4321");
    std::vector<uint8_t> input(4);

    client.infer(input, 0);
  }
  catch (std::exception& e) {
    std::cerr << e.what() << "\n";
  }

  return 0;
}
