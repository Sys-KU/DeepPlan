#pragma once
#include <client/client.h>
#include <util.h>

#include <random>

class Workload {
 public:
  Workload(std::string model_name, int concurrency, int rate,
           int n_requests, int mp_size, EngineType engine_type,
           std::string addr, std::string port);

  void run();

  std::vector<double> result();

  Client client;

  std::vector<double> latencies;
  std::string model_name;
  int concurrency;
  int rate;
  int n_requests;
  int mp_size;
  EngineType engine_type;
  std::string addr;
  std::string port;
};

class ModelLoader {
 public:
  ModelLoader(std::string model_name, int n_models, EngineType engine_type,
              int mp_size, std::string addr, std::string port);

  void run();

  Client client;

  std::string model_name;
  int n_models;
  EngineType engine_type;
  int mp_size;
  std::string addr;
  std::string port;
};
