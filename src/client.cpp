#include <iostream>
#include <getopt.h>

#include <client/workload.h>
#include <util.h>

struct ClientOptions {
  std::string model_name;
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
  {"model",         required_argument,  0,  'm' },
  {"concurrency",   required_argument,  0,  'c' },
  {"rate",          required_argument,  0,  'r' },
  {"mp_size",       required_argument,  0,  's' },
  {"engine",        required_argument,  0,  'e' },
  {0,               0,                  0,   0  }
};

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME --concurrency/-c CONCURRENCY\n"
      "\t\t--rate/-r RATE [--mp_size/-s SLO]\n"
      "\t\t[--engine/-e {in_memory,demand,pipeline,deepplan}]\n",
      program_name);
}

void parseOptions(ClientOptions** benchmark_options, int argc, char** argv) {
  *benchmark_options = new ClientOptions();
  auto options = *benchmark_options;
  char flag;

  char engine_types[][20] = { "in_memory", "demand", "pipeline", "deepplan"};
  int n_types = sizeof(engine_types) / 20;
  bool found = false;

  options->mp_size   = 1;
  options->n_warmup  = 1000;
  options->n_test    = 10000;
  options->engine_type = EngineType::DEEPPLAN;

  while ((flag = getopt_long(argc, argv, "c:e:hm:r:s:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        options->model_name = std::string(optarg);
        break;
      case 'c':
        options->concurrency = (int)strtol(optarg, NULL, 10);
        break;
      case 'r':
        options->rate = (int)strtol(optarg, NULL, 10);
        break;
      case 's':
        options->mp_size = (int)strtol(optarg, NULL, 10);
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
      default:
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
        break;
        bool found = false;
    }
  }
}

void simple(ClientOptions* options) {
  std::string model_name = options->model_name;
  int concurrency = options->concurrency;
  int rate = options->rate;
  int mp_size = options->mp_size;
  EngineType engine_type = options->engine_type;

  int n_warmup = options->n_warmup;
  int n_test = options->n_test;

  auto model_loader = new ModelLoader(model_name, concurrency, engine_type, mp_size,
                                      "127.0.0.1", "4321");

  std::cout << "Upload Model...\n";
  model_loader->run();

  auto warmup = new Workload(model_name, concurrency, rate, n_warmup, mp_size,
                             engine_type, "127.0.0.1", "4321");
  auto workload = new Workload(model_name, concurrency, rate, n_test, mp_size,
                               engine_type, "127.0.0.1", "4321");

  std::cout << "Warmup...\n";
  warmup->run();

  std::cout << "Test...\n";
  workload->run();

  auto latencies = workload->result();

  std::cout << "99% Latency: " << latencies[(latencies.size()*0.99-1)] << " ms\n";
}


int main(int argc, char** argv) {
  ClientOptions* client_options;
  parseOptions(&client_options, argc, argv);

  try {
    simple(client_options);
  }
  catch (std::exception& e) {
    std::cerr << e.what() << "\n";
  }

  return 0;
}
