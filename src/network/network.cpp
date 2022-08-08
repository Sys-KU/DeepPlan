#include <iostream>
#include <network/network.h>

namespace network {

message_receiver::message_receiver(message_connection* conn, message_handler& handler)
  : socket_(conn->get_socket()),
    conn_(conn),
    handler_(handler) {};

void message_receiver::start() {
  read_new_message();
}

void message_receiver::read_new_message() {
  boost::asio::async_read(socket_, boost::asio::buffer(pre_header, 32),
      boost::bind(&message_receiver::handle_pre_hdr_read, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));
}

void message_receiver::handle_pre_hdr_read(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:handle_pre_hdr_read] " << error.message().data() << "\n";
    return;
  }

  boost::asio::async_read(socket_, boost::asio::buffer(header_buf, pre_header[0]),
      boost::bind(&message_receiver::handle_hdr_read, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));
}

void message_receiver::handle_hdr_read(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:hanlde_hdr_read] " << error.message().data() << "\n";
    return;
  }

  res_ = handler_.new_rx_message(pre_header[0], pre_header[1], pre_header[2], pre_header[3]);
  res_->header_received(header_buf, pre_header[0]);

  int64_t body_len = res_->get_rx_body_len();

  if (body_len > 0) {
    boost::asio::async_read(socket_, boost::asio::buffer(res_->rx_body_buf(), body_len),
      boost::bind(&message_receiver::handle_body_read, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));

  }
  else {
    handle_read_end();
  }

}

void message_receiver::handle_read_end() {
  bool is_continue;
  is_continue = handler_.completed_receive(conn_, res_);
  if (!is_continue)
    return;

  res_ = 0;
  read_new_message();
}

void message_receiver::handle_body_read(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:handle_body_read] " << error.message().data() << "\n";
    return;
  }

  res_->body_buf_received(bytes_transferred);

  handle_read_end();
}

message_sender::message_sender(message_connection* conn, message_handler& handler)
  : socket_(conn->get_socket()),
    conn_(conn),
    handler_(handler),
    req_(0) {};

void message_sender::send_message(message_tx& req) {
  tx_queue_.push(&req);
  conn_->io_service_.post(boost::bind(&message_sender::try_send, this));
}

void message_sender::try_send() {
  std::lock_guard<std::mutex> lock(queue_mutex);

  if (!req_) send_next_message();
}

void message_sender::send_next_message() {
  message_tx *req;
  if (!tx_queue_.try_pop(req)) {
    return;
  }
  start_send(*req);
}

void message_sender::start_send(message_tx& req) {
  pre_header[0] = req.get_tx_hdr_len();
  pre_header[1] = req.get_tx_body_len();
  pre_header[2] = req.get_tx_req_id();
  pre_header[3] = req.get_tx_msg_type();

  req.serialize_header(header_buf);

  req_ = &req;

  boost::asio::async_write(socket_, boost::asio::buffer(pre_header, 32),
      boost::bind(&message_sender::handle_pre_hdr_write, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));
}

void message_sender::handle_pre_hdr_write(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:handle_pre_hdr_write] " << error.message().data() << "\n";
    return;
  }

  boost::asio::async_write(socket_, boost::asio::buffer(header_buf, pre_header[0]),
      boost::bind(&message_sender::handle_hdr_write, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred));
}

void message_sender::handle_hdr_write(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:handle_hdr_write] " << error.message().data() << "\n";
    return;
  }

  uint64_t body_len = req_->get_tx_body_len();

  if (body_len > 0) {
    boost::asio::async_write(socket_, boost::asio::buffer(req_->tx_body_buf(), body_len),
        boost::bind(&message_sender::handle_body_write, this,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
  }
  else {
    handle_write_end();
  }
}

void message_sender::handle_write_end() {
  handler_.completed_transmit(conn_, req_);

  std::lock_guard<std::mutex> lock(queue_mutex);
  req_ = 0;
  send_next_message();
}

void message_sender::handle_body_write(const boost::system::error_code& error,
    size_t bytes_transferred) {
  if (error) {
    std::cerr << "[Error:handle_body_write] " << error.message().data() << "\n";
    return;
  }

  handle_write_end();
}

message_connection::message_connection(boost::asio::io_service& io_service, message_handler& handler)
  : socket_(io_service),
    resolver_(io_service),
    io_service_(io_service),
    msg_rx_(message_receiver(this, handler)),
    handler_(handler),
    is_connected(false) {};

boost::asio::ip::tcp::socket& message_connection::get_socket() {
  return socket_;
}

void message_connection::connect(const std::string& server, const std::string& port) {
  boost::asio::ip::tcp::resolver::query query(server, port);
  resolver_.async_resolve(query,
      boost::bind(&message_connection::handle_resolved, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::iterator));

  while (!is_connected.load());
}

void message_connection::established() {
  boost::asio::ip::tcp::no_delay option(true);
  socket_.set_option(option);

  msg_rx_.start();
  ready();
}

void message_connection::ready() {
  is_connected.store(true);
}

void message_connection::handle_resolved(const boost::system::error_code& error,
    boost::asio::ip::tcp::resolver::iterator endpoint_iterator) {
  if (error) {
    std::cerr << "[Error:handle_resolved] " << error.message().data() << "\n";
    return;
  }

  boost::asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
  socket_.async_connect(endpoint,
      boost::bind(&message_connection::handle_established, this,
        boost::asio::placeholders::error));
}

void message_connection::handle_established(const boost::system::error_code& error) {
  if (error) {
    std::cerr << "[Error:handle_established] " << error.message().data() << "\n";
    return;
  }

  established();
}
}

