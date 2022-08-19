#pragma once
#include <client/client.h>
#include <util.h>

#include <random>

struct WorkloadResult {
  double latency_99;
  double cold_rate;
  double goodput_rate;
};

class Workload {
 public:
  Workload(int concurrency, int rate,
           int n_requests, std::string addr, std::string port);

  Workload(std::vector<unsigned>& rates,
           std::string addr, std::string port);

  void run(std::vector<std::vector<char>>& inputs);

  WorkloadResult result(int slo);

  Client client;

  std::vector<std::string> model_names;
  int concurrency;
  int rate;
  int n_requests;
  std::string addr;
  std::string port;

 private:
  std::vector<std::pair<double, int>> _traces;
  std::vector<double> latencies;
  int cold_start_cnt = 0;
};

class ModelLoader {
 public:
  ModelLoader(std::vector<std::string> model_name,
              int n_models, EngineType engine_type,
              int mp_size, std::string addr, std::string port);

  void run();

  Client client;

  std::vector<std::vector<char>> inputs;
  std::vector<std::string> model_names;
  int n_models;
  EngineType engine_type;
  int mp_size;
  std::string addr;
  std::string port;
};
