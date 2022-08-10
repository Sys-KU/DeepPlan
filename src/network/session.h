#pragma once
#include <network/network.h>
#include <network/server_api.h>

#include <atomic>
#include <thread>
#include <future>

#include "deepcache.pb.h"
#include "tbb/concurrent_queue.h"

namespace network {

class Session : public message_connection, message_handler {
 public:
  Session(boost::asio::io_service& io_service)
    : message_connection(io_service, *this),
      msg_tx_(this, *this) {};

  virtual message_rx* new_rx_message(uint64_t hdr_len, uint64_t body_len,
                                     uint64_t req_id, uint64_t msg_type) {};

  virtual bool completed_receive(message_connection* conn, message_rx* req) {};

  virtual void completed_transmit(message_connection* conn, message_tx* req) {};

 protected:
  message_sender msg_tx_;
};

class SrvSession;

struct Message { SrvSession* srv_session; serverapi::Request* req; };

typedef tbb::concurrent_bounded_queue<Message> MessageQueue;

class SrvSession : public Session {
 public:
  SrvSession(boost::asio::io_service& io_service,
             MessageQueue& messages);

  void send_response(serverapi::Response* response);

  message_rx* new_rx_message(uint64_t hdr_len, uint64_t body_len,
                             uint64_t req_id, uint64_t msg_type);

  bool completed_receive(message_connection* conn, message_rx* req);

  void completed_transmit(message_connection* conn, message_tx* req);

 private:
  MessageQueue& messages_;
};

class ClientSession : public Session {
 public:
  ClientSession(boost::asio::io_service& io_service);

  std::future<serverapi::Response*> send_request_async(serverapi::Request& request, std::function<void(serverapi::Response*)> onSuccess);

  serverapi::Response* send_request(serverapi::Request& request, std::function<void(serverapi::Response*)> onSuccess);

  void await_completion();

  message_rx* new_rx_message(uint64_t hdr_len, uint64_t body_len,
                             uint64_t req_id, uint64_t msg_type);

  bool completed_receive(message_connection* conn, message_rx* req);

  void completed_transmit(message_connection* conn, message_tx* req);

 private:
  std::atomic_int request_seed_id;
  std::atomic_int received_rsp_cnt;
  std::map<uint64_t, std::function<void(serverapi::Response*)>> requests;
};

}
