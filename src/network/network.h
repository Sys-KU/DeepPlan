#pragma once

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <atomic>
#include <mutex>
#include "tbb/concurrent_queue.h"

#include <network/message.h>

namespace network {

class message_connection;

class message_handler {
 public:
  virtual message_rx* new_rx_message(
      uint64_t hdr_len,
      uint64_t body_len,
      uint64_t req_id,
      uint64_t msg_type) = 0;

  virtual bool completed_receive(message_connection *conn, message_rx *req) = 0;
  virtual void completed_transmit(message_connection *conn, message_tx *req) = 0;
};

class message_receiver {
 public:
  message_receiver(message_connection* conn, message_handler& handler);

  void start();

  void read_new_message();

 private:
  void handle_pre_hdr_read(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_hdr_read(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_body_read(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_read_end();

  void abort_connection(const char* msg);

  boost::asio::ip::tcp::socket& socket_;

  message_connection* conn_;
  message_handler& handler_;
  message_rx* res_;

  size_t body_left;
  /* header_len, body_len, req_id, message_type */
  uint64_t pre_header[4];
  char header_buf[1024];
};

class message_sender {
 public:
  message_sender(message_connection* conn, message_handler& handler);

  void send_message(message_tx& req);

 private:
  void try_send();

  void send_next_message();

  void start_send(message_tx& req);

  void handle_pre_hdr_write(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_hdr_write(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_body_write(const boost::system::error_code& error,
      size_t bytes_transferred);

  void handle_write_end();

  void abort_connection(const char* msg);

  boost::asio::ip::tcp::socket& socket_;
  message_connection* conn_;
  message_handler& handler_;
  message_tx* req_;
  uint64_t pre_header[4];
  char header_buf[1024];

  std::mutex queue_mutex;
  tbb::concurrent_queue<message_tx*> tx_queue_;
};

class message_connection {
 public:
  message_connection(boost::asio::io_service& io_service, message_handler& handler);

  boost::asio::ip::tcp::socket& get_socket();

  void connect(const std::string& host, const std::string& port);

  void established();

  void abort_connection(const char* msg);

  void close(const char* reason);

  virtual void ready();

 private:
  void handle_resolved(const boost::system::error_code& error,
      boost::asio::ip::tcp::resolver::iterator endpoint_iterator);

  void handle_established(const boost::system::error_code& error);

  boost::asio::ip::tcp::socket socket_;
  boost::asio::ip::tcp::resolver resolver_;
  message_receiver msg_rx_;
  message_handler& handler_;

 protected:
  std::atomic_bool is_connected;

 public:
  boost::asio::io_service& io_service_;
};

}
