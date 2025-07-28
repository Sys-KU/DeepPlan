#include <iostream>
#include <string>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>
#include <time_util.h>
#include <options.h>


void benchmark(BenchmarkOptions options) {
  double t1, t2, total_ms = 0;
  std::vector<double> latencies;

  int num_warmup = options.num_warmup;
  int num_test   = options.num_test;
  int batch_size  = options.batch_size;
  at::Device target_device(at::kCUDA, options.devices[0]);

  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }
  std::string model_path = std::string(model_repo) + "/" + options.model_name;

  torch::NoGradGuard no_grad;

  deepplan::Model* model = new deepplan::Model(
                                            options.model_name,
                                            model_path,
                                            options.engine_type,
                                            options.devices);

  util::InputGenerator input_generator;

  ScriptModuleInput inputs;
  input_generator.generate_input(options.model_name, batch_size, &inputs);

  for (auto& input : inputs) {
    input = input.toTensor().to(model->target_device);
  }

  if (options.engine_type == IN_MEMORY)
    model->to(target_device);

  for (int step = 0; step < num_warmup+num_test; step++) {
    t1 = util::now();

    if (options.engine_type == ON_DEMAND) {
      model->to(target_device, true);
      torch::cuda::synchronize(target_device.index());
    }

    auto outputs = model->forward(inputs);

    t2 = util::now();

    if (options.engine_type != IN_MEMORY) {
      model->clear();
    }

    if (step >= num_warmup) {
      latencies.push_back((t2-t1) / 1e6);
    }
  }

  std::sort(latencies.begin(), latencies.end());

  total_ms = std::accumulate(latencies.begin(), latencies.end(), 0.f);
  double avg_latency = total_ms / num_test;

  std::cout << "Average Latency : " << avg_latency << " ms\n";
  std::cout << "Min Latency : " << latencies.front() << " ms\n";
  std::cout << "Max Latency : " << latencies.back() << " ms\n";

  return;
}

int main(int argc, char** argv) {
  BenchmarkOptions benchmark_options;
  benchmark_options.parseOptions(argc, argv);

  std::cout << "Benchmarking Inference " << benchmark_options.model_name << "\n";

  deepplan::Init();

  benchmark(benchmark_options);

  deepplan::Deinit();

  return 0;
}
