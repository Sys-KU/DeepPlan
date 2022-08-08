#pragma once

#include <network/session.h>
#include <network/message.h>
#include <controller.h>

#include <server_api.h>
#include <boost/asio.hpp>

#include "tbb/concurrent_queue.h"

#define DEFAULT_PORT 4321

class Server {
 public:
  Server(std::string model_repo, int port);

  void run();

  void send_response(serverapi::Response* response);

 private:
  void start_accept();

  void handle_accept(network::SrvSession* new_session, const boost::system::error_code& error);

  boost::asio::io_service io_service_;
  boost::asio::ip::tcp::acceptor acceptor_;

  network::MessageQueue messages;

  network::SrvSession* current_session;

  Controller* controller;
};
