#include <iostream>
#include <string>
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
  EngineType engine_type;
  std::vector<int> devices;
  int batch_size;
  int num_warmup;
  int num_test;
};

static struct option long_options[] =
{
  {"help",    no_argument,       0, 'h' },
  {"model",   required_argument, 0, 'm' },
  {"engine",  required_argument, 0, 'e' },
  {"devices", required_argument, 0, 'd' },
  {"batch",   required_argument, 0, 'b' },
  {0,         0,                 0,  0  }
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME [--device/-d DEVICES [DEVICES ...]]\n"
      "\t\t[--engine/-e {in_memory,demand,pipeline,deepplan}]\n"
      "\t\t[--batch/-b BATCH_SIZE",
      program_name);
}

void parseOptions(BenchmarkOptions** benchmark_options, int argc, char** argv) {
  *benchmark_options = new BenchmarkOptions();
  auto options = *benchmark_options;
  char flag;

  char engine_types[][20] = { "in_memory", "demand", "pipeline", "deepplan"};
  int n_types = sizeof(engine_types) / 20;
  bool found = false;

  options->num_warmup  = 20;
  options->num_test    = 200;
  options->batch_size  = 1;
  options->engine_type = EngineType::IN_MEMORY;
  options->devices     = std::vector<int>(1, 0); // = [0]

  while ((flag = getopt_long(argc, argv, "b:d:e:hm:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        options->model_name = std::string(optarg);
        break;
      case 'e':
        found = false;
        for (int i = 0; i < n_types; i++) {
          if (!strcmp(engine_types[i], optarg)) {
            options->engine_type = EngineType(i);
            found = true;
            break;
          }
        }

        if (!found) {
          print_usage(argv[0]);
          fprintf(stderr, "[Error] argument --engine/-e: invalid choice: %s (choose from",
              optarg);
          for (int i = 0; i < n_types; i++) {
            fprintf(stderr, " \'%s\'", engine_types[i]);
          }
          fprintf(stderr, ")\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'b':
        options->batch_size = (int)strtol(optarg, NULL, 10);
        break;
      case 'd':
        optind--;
        {
          std::vector<int> devices;
          for ( ; optind < argc && *argv[optind] != '-'; optind++) {
            devices.push_back((int)strtol(argv[optind], NULL, 10));
          }
          options->devices = devices;
        }
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
  double t1, t2, total_ms = 0;
  int num_warmup = options->num_warmup;
  int num_test   = options->num_test;
  int batch_size  = options->batch_size;
  at::Device target_device(at::kCUDA, options->devices[0]);

  torch::NoGradGuard no_grad;

  deepplan::Model* model = new deepplan::Model(
                                            options->model_name,
                                            options->engine_type,
                                            options->devices);

  util::InputGenerator input_generator;

  ScriptModuleInput inputs;
  input_generator.generate_input(options->model_name, batch_size, &inputs);

  for (auto& input : inputs) {
    input = input.toTensor().to(model->target_device);
  }

  if (options->engine_type == IN_MEMORY)
    model->to(target_device);

  for (int step = 0; step < num_warmup+num_test; step++) {
    t1 = util::now();

    if (options->engine_type == ON_DEMAND) {
      model->to(target_device, true);
      torch::cuda::synchronize(target_device.index());
    }

    auto outputs = model->forward(inputs);

    torch::cuda::synchronize(target_device.index());
    t2 = util::now();

    if (options->engine_type != IN_MEMORY) {
      model->clear();
    }

    if (step >= num_warmup) {
      total_ms += ((t2-t1) / 1e6);
    }
  }

  double avg_latency = total_ms / num_test;
  std::cout << "Average Latency : " << avg_latency << " ms\n";

  return;
}

int main(int argc, char** argv) {
  BenchmarkOptions* benchmark_options;
  parseOptions(&benchmark_options, argc, argv);

  std::cout << "Benchmarking Inference " << benchmark_options->model_name << "\n";

  deepplan::Init();

  benchmark(benchmark_options);

  deepplan::Deinit();

  return 0;
}
