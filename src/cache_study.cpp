#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <deepcache/model.h>
#include <deepcache/engine.h>
#include <util.h>

struct BenchmarkOptions {
  std::string model_name;
  int batch_size;
  int num_warmup;
  int num_test;
};

static struct option long_options[] =
{
  {"help",    no_argument,       0, 'h' },
  {"model",   required_argument, 0, 'm' },
  {"batch",   required_argument, 0, 'b' },
  {0,         0,                 0,  0  }
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME\n"
      "\t\t[--batch/-b BATCH_SIZE\n",
      program_name);
}

void parseOptions(BenchmarkOptions** benchmark_options, int argc, char** argv) {
  *benchmark_options = new BenchmarkOptions();
  auto options = *benchmark_options;
  char flag;

  bool found = false;

  options->num_warmup  = 20;
  options->num_test    = 30;
  options->batch_size  = 1;

  while ((flag = getopt_long(argc, argv, "b:hm:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        options->model_name = std::string(optarg);
        break;
      default:
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
        break;
        bool found = false;
    }
  }
}

void benchmark(BenchmarkOptions* options) {
  int num_warmup = options->num_warmup;
  int num_test   = options->num_test;
  int batch_size  = options->batch_size;
  at::Device target_device(at::kCUDA, 0);

  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }

  std::string model_path = std::string(model_repo) + "/" + options->model_name;

  torch::NoGradGuard no_grad;

  deepcache::Model* model = new deepcache::Model(
                                            options->model_name,
                                            model_path,
                                            target_device.index()
                                          );

  util::InputGenerator input_generator;

  ScriptModuleInput inputs;
  input_generator.generate_input(options->model_name, batch_size, &inputs);

  for (auto& input : inputs) {
    input = input.toTensor().to(model->target_device);
  }

  double t1, t2, total_ms, avg_latency;
  size_t load_size;
  std::cout << "Number of cached layers, Latency (ms), Load Size (MB)\n";
  for (int i = 0; i <= model->n_layers; i++) {
    total_ms = 0;

    for (int step = 0; step < num_warmup+num_test; step++) {
      model->load_layers(i);

      // TODO: remove recalculatation
      load_size = util::getModuleSize(model->model, true);

      t1 = util::now();

      auto outputs = model->forward(inputs);

      torch::cuda::synchronize(target_device.index());
      t2 = util::now();

      if (step >= num_warmup) {
        total_ms += ((t2-t1) / 1e6);
      }

      model->clear();
    }

    avg_latency = total_ms / num_test;

    std::cout << i << ", " << avg_latency << ", "
              << load_size / 1024.f / 1024.f << "\n";
  }

  return;
}

int main(int argc, char** argv) {
  BenchmarkOptions* benchmark_options;
  parseOptions(&benchmark_options, argc, argv);

  std::cout << "Caching Study " << benchmark_options->model_name << "\n";

  deepcache::Init();

  benchmark(benchmark_options);

  deepcache::Deinit();

  return 0;
}
