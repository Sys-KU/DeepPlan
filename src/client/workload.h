#pragma once
#include <client/client.h>
#include <util.h>

struct WorkloadResult {
  double avg_infer_time;
  double throughput;
  double latency_50;
  double latency_99;
  double cold_rate;
  double goodput_rs;
  double goodput_rate;
};

class Workload {
 public:
  Workload(int concurrency, int rate, int n_requests,
           std::string dist_type, bool disable_timeout,
           std::string addr, std::string port);

  Workload(int concurrency, int rate, int n_requests,
           float alpha, bool disable_timeout,
           std::string addr, std::string port);

  Workload(std::vector<unsigned>& rates, bool disable_timeout,
           std::string addr, std::string port);

  void run(std::vector<std::vector<char>>& inputs);

  WorkloadResult result();

  void dump(std::ofstream ofs);

  Client client;

  std::vector<std::string> model_names;
  int concurrency;
  int rate;
  int n_requests;
  bool disable_timeout;
  std::string addr;
  std::string port;

 private:
  std::vector<std::pair<double, int>> _traces;
  double elapsed_time;

  struct ResResult {
    ResResult(const int model_id, const double latency, const double infer_time,
              const bool good)
      : model_id(model_id), latency(latency), infer_time(infer_time),
        good(good) {};

    const int model_id;
    const double latency;
    const double infer_time;
    const bool good;
  };
  std::vector<ResResult> res_results;
  int cold_start_cnt = 0;
  int timeout_cnt = 0;
};

class ModelLoader {
 public:
  ModelLoader(std::vector<std::string> model_names, int n_models,
              EngineType engine_type, ReclaimPolicy r_policy,
              int mp_size, bool disable_prefetch, std::string addr,
              std::string port);

  void run();

  Client client;

  std::vector<std::vector<char>> inputs;
  std::vector<std::string> model_names;
  int n_models;
  EngineType engine_type;
  ReclaimPolicy r_policy;
  int mp_size;
  bool disable_prefetch;
  std::string addr;
  std::string port;
};
