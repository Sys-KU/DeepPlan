#pragma once
#include <client/client.h>
#include <util.h>

#include <random>

struct WorkloadResult {
  double throughput;
  double latency_50;
  double latency_99;
  double cold_rate;
  double goodput_rate;
};

class Workload {
 public:
  Workload(int concurrency, int rate, int n_requests,
           std::string dist_type, std::string addr, std::string port);

  Workload(int concurrency, int rate, int n_requests,
           float alpha, std::string addr, std::string port);

  Workload(std::vector<unsigned>& rates,
           std::string addr, std::string port);

  void run(std::vector<std::vector<char>>& inputs);

  WorkloadResult result(int slo);

  void dump(std::string dump_file);

  Client client;

  std::vector<std::string> model_names;
  int concurrency;
  int rate;
  int n_requests;
  std::string addr;
  std::string port;

 private:
  std::vector<std::pair<double, int>> _traces;
  double elapsed_time;

  struct ResResult {
    ResResult(const int model_id, const double latency, const double infer_time)
      : model_id(model_id), latency(latency), infer_time(infer_time) {};

    const int model_id;
    const double latency;
    const double infer_time;
  };
  std::vector<ResResult> res_results;
  int cold_start_cnt = 0;
};

class ModelLoader {
 public:
  ModelLoader(std::vector<std::string> model_names, int n_models,
              EngineType engine_type, ReclaimPolicy r_policy,
              int mp_size, int slo_ms, std::string addr, std::string port);

  void run();

  Client client;

  std::vector<std::vector<char>> inputs;
  std::vector<std::string> model_names;
  int n_models;
  EngineType engine_type;
  ReclaimPolicy r_policy;
  int mp_size;
  int slo_ms;
  std::string addr;
  std::string port;
};
