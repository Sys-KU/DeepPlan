#include <iostream>
#include <getopt.h>

#include <client/workload.h>
#include <client/azure.h>
#include <util.h>

typedef enum {
  SIMPLE = 0,
  BURSTY,
  AZURE,
} WorkloadType;

struct ClientOptions {
  WorkloadType workload_type;
  std::vector<std::string> model_names;
  int concurrency;
  int rate;
  int mp_size;
  EngineType engine_type;
  int slo;
  int n_warmup;
  int n_test;
};

static struct option long_options[] =
{
  {"help",          no_argument,        0,  'h' },
  {"workload",      required_argument,  0,  'w' },
  {"model",         required_argument,  0,  'm' },
  {"concurrency",   required_argument,  0,  'c' },
  {"rate",          required_argument,  0,  'r' },
  {"mp_size",       required_argument,  0,  'p' },
  {"engine",        required_argument,  0,  'e' },
  {"slo",           required_argument,  0,  's' },
  {0, 0, 0, 0}
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --workload/-w WORKLOAD --model/-m MODEL_NAME\n"
      "\t\t--concurrency/-c CONCURRENCY --rate/-r RATE [--mp_size/-p MP_SIZE]\n"
      "\t\t[--engine/-e {in_memory,demand,pipeline,deepplan}]\n"
      "\t\t[--slo/-s SLO]\n",
      program_name);
}

void parseOptions(ClientOptions** benchmark_options, int argc, char** argv) {
  *benchmark_options = new ClientOptions();
  auto options = *benchmark_options;
  char flag;

  char engine_types[][20] = { "in_memory", "demand", "pipeline", "deepplan" };
  char workload_types[][20] = { "simple", "bursty", "azure" };
  int n_engine_types = sizeof(engine_types) / 20;
  int n_workload_types = sizeof(workload_types) / 20;
  bool found = false;

  options->mp_size   = 1;
  options->n_warmup  = 1000;
  options->n_test    = 10000;
  options->engine_type = EngineType::DEEPPLAN;
  options->slo       = 100;

  while ((flag = getopt_long(argc, argv, "c:e:hm:r:s:w:p:", long_options, NULL)) != -1) {
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        optind--;
        {
          std::vector<std::string> model_names;
          for ( ; optind < argc && *argv[optind] != '-'; optind++) {
            model_names.push_back(std::string(argv[optind]));
          }
          options->model_names = model_names;
        }
        break;
      case 'c':
        options->concurrency = (int)strtol(optarg, NULL, 10);
        break;
      case 'r':
        options->rate = (int)strtol(optarg, NULL, 10);
        break;
      case 'p':
        options->mp_size = (int)strtol(optarg, NULL, 10);
        break;
      case 's':
        options->slo = (int)strtol(optarg, NULL, 10);
        break;
      case 'e':
        found = false;
        for (int i = 0; i < n_engine_types; i++) {
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
          for (int i = 0; i < n_engine_types; i++) {
            fprintf(stderr, " \'%s\'", engine_types[i]);
          }
          fprintf(stderr, ")\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'w':
        found = false;
        for (int i = 0; i < n_workload_types; i++) {
          if (!strcmp(workload_types[i], optarg)) {
            options->workload_type = WorkloadType(i);
            found = true;
            break;
          }
        }

        if (!found) {
          print_usage(argv[0]);
          fprintf(stderr, "[Error] argument --workload/-w: invalid choice: %s (choose from",
              optarg);
          for (int i = 0; i < n_workload_types; i++) {
            fprintf(stderr, " \'%s\'", workload_types[i]);
          }
          fprintf(stderr, ")\n");
          exit(EXIT_FAILURE);
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

void simple_experiment(ClientOptions* options) {
  std::vector<std::string> model_names = options->model_names;
  int concurrency = options->concurrency;
  int rate = options->rate;
  int mp_size = options->mp_size;
  EngineType engine_type = options->engine_type;
  int slo = options->slo;

  int n_warmup = options->n_warmup;
  int n_test = rate * 100;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type, mp_size,
                                      "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  auto warmup = new Workload(concurrency, rate, n_warmup, "127.0.0.1", "4321");
  auto workload = new Workload(concurrency, rate, n_test, "127.0.0.1", "4321");

  std::cout << "Warmup...\n";
  warmup->run(model_loader->inputs);

  std::cout << "Test...\n";
  workload->run(model_loader->inputs);

  auto result = workload->result(slo);

  std::cout << "99% Latency: " << result.latency_99 << " ms\n";
  std::cout << "Cold Start Rate: " << result.cold_rate << " %\n";
  std::cout << "Goodput Rate: " << result.goodput_rate << " %\n";
}

void bursty_experiment(ClientOptions* options) {
  std::vector<std::string> model_names = options->model_names;
  int concurrency = options->concurrency;
  int rate = options->rate;
  int mp_size = options->mp_size;
  int slo = options->slo;
  EngineType engine_type = options->engine_type;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type, mp_size,
                                      "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  std::vector<Workload*> warmups;
  std::vector<Workload*> workloads;
  for (int i = 1; i <= concurrency; i++) {
    warmups.push_back(new Workload(i, rate, rate, "127.0.0.1", "4321"));

    workloads.push_back(new Workload(i, rate, rate, "127.0.0.1", "4321"));
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

void azure_experiment(ClientOptions* options) {
  std::vector<std::string> model_names = options->model_names;
  int concurrency = options->concurrency;
  int rate = options->rate;
  int mp_size = options->mp_size;
  EngineType engine_type = options->engine_type;
  int slo = options->slo;

  auto model_loader = new ModelLoader(model_names, concurrency, engine_type, mp_size,
                                      "127.0.0.1", "4321");

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
  ClientOptions* client_options;
  parseOptions(&client_options, argc, argv);

  try {
    switch (client_options->workload_type) {
      case WorkloadType::SIMPLE:
        simple_experiment(client_options);
        break;
      case WorkloadType::BURSTY:
        bursty_experiment(client_options);
        break;
      case WorkloadType::AZURE:
        azure_experiment(client_options);
        break;
    }
  }
  catch (std::exception& e) {
    std::cerr << e.what() << "\n";
  }

  return 0;
}
