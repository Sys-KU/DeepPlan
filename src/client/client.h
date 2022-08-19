#pragma once

#include <network/session.h>
#include <util.h>
#include <atomic>

class Client {
 public:
  Client();

  ~Client();

  void connect(const std::string& srv_ip, const std::string& port);

  void run();

  void infer_async(std::vector<char>& input, int model_id,
                   std::function<void(serverapi::Response* rsp)> onSuccess);

  serverapi::UploadModelResponse* upload_model(std::vector<std::string> model_name, int n_models, EngineType engine_type, int mp_size);

  void close();

  void sync_close();

  void shutdown();

private:
  std::atomic_bool alive;
  std::thread network_thr;
  boost::asio::io_service io_service_;
  network::ClientSession* session;
};
