#pragma once

#include <network/session.h>
#include <atomic>

class Client {
 public:
  Client();

  ~Client();

  void connect(const std::string& srv_ip, const std::string& port);

  void run();

  void close();

  void sync_close();

  void infer(std::vector<uint8_t>& input, int model_id);

  void infer(std::vector<uint8_t>& input, int model_id, std::function<void(void)> onSuccess);

  void shutdown();

private:
  std::atomic_bool alive;
  std::thread network_thr;
  boost::asio::io_service io_service_;
  network::ClientSession* session;
};
