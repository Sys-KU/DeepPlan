#include <server/server.h>
#include <network/session.h>

#include <boost/asio.hpp>

Server::Server(int port)
  : io_service_(),
    acceptor_(io_service_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
    alive(false) {};

Server::~Server() {
  shutdown();
}

void Server::run() {
  controller = new Controller(messages);

  start_accept();

  alive = true;

  std::cout << "Server Ready\n";
  io_service_.run();
}

void Server::shutdown() {
  if (alive) {
    std::cout << "Closing Server\n";
    alive = false;
    controller->shutdown();

    boost::system::error_code ec;
    acceptor_.close(ec);
    if (ec){
      std::cerr << "Acceptor Error occured\n";
    }
    // If connecting session, close the session.
  }
}

void Server::send_response(serverapi::Response* response) {
  current_session->send_response(response);
}

void Server::start_accept() {
  network::SrvSession* new_session = new network::SrvSession(io_service_, messages);
  acceptor_.async_accept(new_session->get_socket(),
      boost::bind(&Server::handle_accept, this, new_session,
        boost::asio::placeholders::error));
}

void Server::handle_accept(network::SrvSession* new_session,
    const boost::system::error_code& error) {
  if (error) {
    std::cerr << "[Error] " << error.message() << std::endl;
    delete new_session;
    return;
  }

  // FIXME: Should enable to handle multi client
  new_session->established();
  current_session = new_session;

  // wait when the connection from the client is disconnected.
  start_accept();
}
