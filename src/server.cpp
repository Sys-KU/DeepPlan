#include <server/server.h>
#include <signal.h>
#include <options.h>


class InterruptException : public std::exception
{
  public:
    InterruptException(int s) : S(s) {}
    int S;
};

void sig_to_exception(int s)
{
  throw InterruptException(s);
}

int main(int argc, char** argv) {
  ServerOptions server_options;
  server_options.parseOptions(argc, argv);

  {
    // setupt handling interrupt
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sig_to_exception;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
  }

  Server* server;
  try {
    server = new Server(DEFAULT_PORT, server_options);
    server->run();
  }
  catch(InterruptException& e) {
    server->shutdown();
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
