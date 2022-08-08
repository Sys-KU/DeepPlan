#include <server.h>
#include <network/session.h>

#include <thread>
#include <boost/asio.hpp>

Server::Server(std::string model_repo, int port)
  : io_service_(),
    acceptor_(io_service_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
    controller(new Controller(messages, EngineType::DEEPPLAN, 1)) {};

void Server::run() {
  start_accept();

  std::cout << "Server Ready\n";
  io_service_.run();
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

  std::cout << "Accept!\n";

  // FIXME: Should enable to handle multi client
  new_session->established();
  current_session = new_session;

  // wait when the connection from the client is disconnected.
  start_accept();
}

int main(int argc, char** argv) {
  std::string model_repo = "/home/jinu/DeepCache/plans/A5000";

  try {
  Server* server = new Server(model_repo, DEFAULT_PORT);
  server->run();
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
