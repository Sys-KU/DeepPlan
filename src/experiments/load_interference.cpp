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

#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>

struct BenchmarkOptions {
  std::string model_name;
  int batch_size;
  int load_layers;
  int num_warmup;
  int num_test;
};

static struct option long_options[] =
{
  {"help",    no_argument,       0, 'h' },
  {"model",   required_argument, 0, 'm' },
  {"batch",   required_argument, 0, 'b' },
  {"load",    required_argument, 0, 'l' },
  {0,         0,                 0,  0  }
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME\n"
      "\t\t[--batch/-b BATCH_SIZE] [--load/-l LOAD_LAYERS]\n",
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
  options->load_layers = -1;

  while ((flag = getopt_long(argc, argv, "b:hm:l:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        options->model_name = std::string(optarg);
        break;
      case 'b':
        options->batch_size = strtoul(optarg, NULL, 10);
        break;
      case 'l':
        options->load_layers = strtoul(optarg, NULL, 10);
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

void benchmark(BenchmarkOptions* options) {
  int num_warmup = options->num_warmup;
  int num_test   = options->num_test;
  int batch_size  = options->batch_size;
  int load_layers = options->load_layers;

  at::Device target_device(at::kCUDA, 0);

  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }

  std::string model_path = std::string(model_repo) + "/" + options->model_name;

  torch::NoGradGuard no_grad;

  deepplan::Model* model = new deepplan::Model(
                                            options->model_name,
                                            model_path,
                                            EngineType::PIPESWITCH,
                                            {target_device.index()}
                                          );

  deepplan::Model* load_model = new deepplan::Model(
                                              options->model_name,
                                              model_path,
                                              EngineType::PIPESWITCH,
                                              {target_device.index()}
                                            );

  util::InputGenerator input_generator;

  ScriptModuleInput inputs;
  input_generator.generate_input(options->model_name, batch_size, &inputs);

  for (auto& input : inputs) {
    input = input.toTensor().to(model->target_device);
  }

  double t1, t2, total_infer_ms, avg_infer_ms;

  total_infer_ms = 0;

  model->to(model->target_device);

  for (int step = 0; step < num_warmup+num_test; step++) {
    if (load_layers != -1) {
      load_model->load_layers(load_layers, false);
      LoadLayers(load_model);
    }
    t1 = util::now();

    auto outputs = model->forward(inputs);

    torch::cuda::synchronize(target_device.index());
    t2 = util::now();

    if (load_layers != -1) {
      load_model->clear();
    }

    if (step >= num_warmup) {
      total_infer_ms += ((t2-t1) / 1e6);
    }
  }

  avg_infer_ms = total_infer_ms / num_test;

  std::cout << "Average Inference Time : " << avg_infer_ms << " ms\n";
  return;
}

int main(int argc, char** argv) {
  BenchmarkOptions* benchmark_options;
  parseOptions(&benchmark_options, argc, argv);

  std::cout << "Caching Study " << benchmark_options->model_name << " "
            << benchmark_options->batch_size << "-Batch\n";

  deepplan::Init();

  benchmark(benchmark_options);

  deepplan::Deinit();

  return 0;
}
