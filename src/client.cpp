#include <iostream>

#include <client/workload.h>
#include <client/azure.h>
#include <util.h>
#include <options.h>


void simple_experiment(ClientOptions options, std::string dist_type) {
  std::vector<std::string> model_names = options.model_names;
  int concurrency = options.concurrency;
  int rate = options.rate;
  int mp_size = options.mp_size;
  EngineType engine_type = options.engine_type;
  ReclaimPolicy r_policy = options.r_policy;
  int slo = options.slo;
  float alpha = options.alpha;

  int n_warmup = options.n_warmup;
  int n_test = rate * 100;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type,
                                      r_policy, mp_size, "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  if (dist_type == "uniform") {
    alpha = 0.f;
  }

  auto warmup = new Workload(concurrency, rate, n_warmup, alpha, "127.0.0.1", "4321");
  auto workload = new Workload(concurrency, rate, n_test, alpha, "127.0.0.1", "4321");

  std::cout << "Warmup...\n";
  warmup->run(model_loader->inputs);

  std::cout << "Test...\n";
  workload->run(model_loader->inputs);

  auto result = workload->result(slo);

  std::cout << "=======================================\n";
  std::cout << "Throughput: " << result.throughput << " r/s\n";
  std::cout << "50% Latency: " << result.latency_50 << " ms\n";
  std::cout << "99% Latency: " << result.latency_99 << " ms\n";
  std::cout << "Cold Start Rate: " << result.cold_rate << " %\n";
  std::cout << "Goodput Rate: " << result.goodput_rate << " %\n";
  std::cout << "=======================================\n";

  if (!options.dump.empty()) {
    workload->dump(options.dump);
  }
}

void bursty_experiment(ClientOptions options) {
  std::vector<std::string> model_names = options.model_names;
  int concurrency = options.concurrency;
  int rate = options.rate;
  int mp_size = options.mp_size;
  int slo = options.slo;
  EngineType engine_type = options.engine_type;
  ReclaimPolicy r_policy = options.r_policy;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type,
                                      r_policy, mp_size, "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  std::vector<Workload*> warmups;
  std::vector<Workload*> workloads;
  for (int i = 1; i <= concurrency; i++) {
    warmups.push_back(new Workload(i, rate, rate, "uniform", "127.0.0.1", "4321"));

    workloads.push_back(new Workload(i, rate, rate, "uniform", "127.0.0.1", "4321"));
  }

  std::cout << "Bursty Experiment\n";
  std::cout << "Concurrency, 99% Latecny(ms), Cold Start Rate(%), Goodput Rate(%)\n";
  for (int i = 0; i < concurrency; i++) {
    warmups[i]->run(model_loader->inputs);
    workloads[i]->run(model_loader->inputs);
    auto result = workloads[i]->result(slo);

    std::cout << i+1 << ", ";
    std::cout << result.latency_99 << ", ";
    std::cout << result.cold_rate << ", ";
    std::cout << result.goodput_rate << "\n";
  }
}

void azure_experiment(ClientOptions options) {
  std::vector<std::string> model_names = options.model_names;
  int concurrency = options.concurrency;
  int rate = options.rate;
  int mp_size = options.mp_size;
  EngineType engine_type = options.engine_type;
  ReclaimPolicy r_policy = options.r_policy;

  int slo = options.slo;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type,
                                      r_policy, mp_size, "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  auto scaled_traces = azure::load_scaled_trace(rate, concurrency, 2);

  azure::transpose(scaled_traces);

  int period = 180;
  std::vector<Workload*> workloads;
  for (int p = 0; p < period; p++) {
    workloads.push_back(new Workload(scaled_traces[p], "127.0.0.1", "4321"));
  }

  std::cout << "Azure Experiment\n";
  std::cout << "Minutes, Offered Load, 99% Latecny(ms), Cold Start Rate(%), Goodput Rate(%)\n";
  for (int p = 0; p < period; p++) {
    workloads[p]->run(model_loader->inputs);
    auto result = workloads[p]->result(slo);

    std::cout << p << ", ";
    std::cout << workloads[p]->n_requests << ", ";
    std::cout << result.latency_99 << ", ";
    std::cout << result.cold_rate << ", ";
    std::cout << result.goodput_rate << "\n";
  }

}


int main(int argc, char** argv) {
  ClientOptions client_options;
  client_options.parseOptions(argc, argv);

  try {
    switch (client_options.workload_type) {
      case ClientOptions::WorkloadType::SIMPLE_UNIFORM:
        simple_experiment(client_options, "uniform");
        break;
      case ClientOptions::WorkloadType::SIMPLE_ZIPF:
        simple_experiment(client_options, "zipfian");
        break;
      case ClientOptions::WorkloadType::BURSTY:
        bursty_experiment(client_options);
        break;
      case ClientOptions::WorkloadType::AZURE:
        azure_experiment(client_options);
        break;
    }
  }
  catch (std::exception& e) {
    std::cerr << e.what() << "\n";
  }

  return 0;
}
