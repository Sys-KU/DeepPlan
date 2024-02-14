#include <iostream>
#include <string>
#include <stack>
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
  bool verbose;
  int batch_size;
  int num_warmup;
  int num_test;
};

static struct option long_options[] =
{
  {"verbose", no_argument,       0, 'v' },
  {"help",    no_argument,       0, 'h' },
  {"model",   required_argument, 0, 'm' },
  {"batch",   required_argument, 0, 'b' },
  {0,         0,                 0,  0  }
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME\n"
      "\t\t[--batch/-b BATCH_SIZE] [--verbose/-v]\n",
      program_name);
}

void parseOptions(BenchmarkOptions** benchmark_options, int argc, char** argv) {
  *benchmark_options = new BenchmarkOptions();
  auto options = *benchmark_options;
  char flag;

  bool found = false;
  bool pass_model = false;

  options->num_warmup  = 20;
  options->num_test    = 30;
  options->batch_size  = 1;
  options->verbose     = false;

  while ((flag = getopt_long(argc, argv, "b:hm:v:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        pass_model = true;
        options->model_name = std::string(optarg);
        break;
      case 'b':
        options->batch_size = strtoul(optarg, NULL, 10);
        break;
      case 'v':
        options->verbose = true;
        break;
      default:
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
        break;
        bool found = false;
    }
  }

  if (!pass_model) {
    print_usage(argv[0]);
    fprintf(stderr, "[Error] the following arguments are required: --model_name/-m\n");
    exit(EXIT_FAILURE);
  }
}

struct InferResult {
  InferResult(double latency, size_t load_size, int n_caches)
    : latency(latency),
      load_size(load_size),
      n_caches(n_caches) {};
  double latency;
  size_t load_size;
  int n_caches;
};

void benchmark(BenchmarkOptions* options) {
  int num_warmup = options->num_warmup;
  int num_test   = options->num_test;
  int batch_size  = options->batch_size;
  bool verbose_flag = options->verbose;

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

  double t1, t2, total_infer_ms, avg_infer_ms, total_load_ms, avg_load_ms;
  std::stack<InferResult> results;
  size_t load_size;
  util::progressbar progressbar;

  if (verbose_flag) {
    std::cout << "Number of cached layers, Load Size (MB), Inference Latency (ms), Load Latency (ms)\n";
  }
  else {
    progressbar = util::progressbar(model->n_layers + 1);
  }

  for (int i = 0; i <= model->n_layers; i++) {
    total_infer_ms = 0;
    total_load_ms = 0;

    for (int step = 0; step < num_warmup+num_test; step++) {
      model->load_layers(i);

      // TODO: remove recalculatation
      load_size = util::getModuleSize(model->model, true);

      t1 = util::now();

      auto outputs = model->forward(inputs);

      torch::cuda::synchronize(target_device.index());
      t2 = util::now();

      if (step >= num_warmup) {
        total_infer_ms += ((t2-t1) / 1e6);
      }

      if (verbose_flag) {
        model->clear();
        model->load_layers(i);

        t1 = util::now();
        model->load_layers(model->n_layers-i, true);

        torch::cuda::synchronize(target_device.index());
        t2 = util::now();

        if (step >= num_warmup) {
          total_load_ms += ((t2-t1) / 1e6);
        }

      }

      model->clear();
    }

    avg_infer_ms = total_infer_ms / num_test;
    avg_load_ms = total_load_ms / num_test;
    results.emplace(avg_infer_ms, load_size, i);

    if (verbose_flag) {
      std::cout << i << ", " << load_size/1024.f/1024.f << ", "
                << avg_infer_ms << ", " << avg_load_ms << "\n";
    }
    else {
        progressbar.update();
    }
  }

  double inmemory_lat = results.top().latency;
  double threshold = inmemory_lat * 1.1;

  InferResult opt_point = results.top();
  while (!results.empty()) {
    auto ret = results.top();
    if (threshold < ret.latency) {
      break;
    }
    opt_point = ret;
    results.pop();
  }

  std::cout << "============ Summary ============\n";
  std::cout << "Total Number of Layers : " << model->n_layers << "\n";
  std::cout << "Total Model Size : " << model->model_size/1024.f/1024.f << " MB\n";
  std::cout << "In-Memory Inference Time : " << inmemory_lat << " ms\n";
  std::cout << "Optimal Point Number of Cached layers : " << opt_point.n_caches << "\n";
  std::cout << "Optimal Point Load Size : " << opt_point.load_size/1024.f/1024.f << " MB\n";
  std::cout << "Optimal Point Inference Time : " << opt_point.latency << " ms\n";
  return;
}

int main(int argc, char** argv) {
  BenchmarkOptions* benchmark_options;
  parseOptions(&benchmark_options, argc, argv);

  std::cout << "Caching Study " << benchmark_options->model_name << " "
            << benchmark_options->batch_size << "-Batch\n";

  deepcache::Init();

  benchmark(benchmark_options);

  deepcache::Deinit();

  return 0;
}
