#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <options.h>
#include <util.h>

void ServerOptions::print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] [--watermark/-w WATERMARK]\n",
      program_name);
}


void ServerOptions::parseOptions(int argc, char** argv) {
  struct option long_options[] =
  {
    {"watermark", no_argument,       0, 'w' },
    {"help",      no_argument,       0, 'h' },
    {0,           0,                 0,  0  }
  };

  char flag;
  bool found = false;

  this->watermark = 0.95f;

  while ((flag = getopt_long(argc, argv, "hw:", long_options, NULL)) != -1) {
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'w':
        this->watermark = strtof(optarg, NULL);
        break;
      default:
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
        break;
        bool found = false;
    }
  }
}


void ClientOptions::print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --workload/-w WORKLOAD --model/-m MODEL_NAME\n"
      "\t\t--concurrency/-c CONCURRENCY --rate/-r RATE [--mp_size/-p MP_SIZE]\n"
      "\t\t[--engine/-e {in_memory,demand,pipeline,deepplan,deepcache}]\n"
      "\t\t[--r_policy {rr, balance}]\n"
      "\t\t[--slo/-s SLO] [--dump/-d]\n",
      program_name);
}


void ClientOptions::parseOptions(int argc, char** argv) {
  struct option long_options[] =
  {
    {"help",          no_argument,        0,  'h' },
    {"workload",      required_argument,  0,  'w' },
    {"model",         required_argument,  0,  'm' },
    {"concurrency",   required_argument,  0,  'c' },
    {"rate",          required_argument,  0,  'r' },
    {"mp_size",       required_argument,  0,  'p' },
    {"engine",        required_argument,  0,  'e' },
    {"r_policy",      required_argument,  0,   0  },
    {"slo",           required_argument,  0,  's' },
    {"dump",          required_argument,  0,  'd' },
    {0, 0, 0, 0}
  };

  char flag;
  int option_index = 0;

  char engine_types[][20] = { "in_memory", "demand", "pipeline", "deepplan", "deepcache" };
  char workload_types[][20] = { "simple_uniform", "simple_zipf", "bursty", "azure", "skew" };
  char r_policies[][20] = { "lru", "rr", "balance" };
  int n_engine_types = sizeof(engine_types) / 20;
  int n_r_polices = sizeof(r_policies) / 20;
  int n_workload_types = sizeof(workload_types) / 20;
  bool found = false;
  bool pass_model = false;
  bool pass_concurrency = false;
  bool pass_rate = false;

  this->mp_size   = 1;
  this->n_warmup  = 1000;
  this->n_test    = 10000;
  this->engine_type = EngineType::DEEPPLAN;
  this->r_policy  = ReclaimPolicy::RR;
  this->slo       = 100;
  this->dump      = "";

  while ((flag = getopt_long(argc, argv, "c:d:e:hm:r:s:w:p:", long_options, &option_index)) != -1) {
    switch (flag) {
      case 0:
        if (long_options[option_index].flag != 0)
          break;
        if (long_options[option_index].name == "r_policy") {
          found = false;
          for (int i = 0; i < n_r_polices; i++) {
            if (!strcmp(r_policies[i], optarg)) {
              this->r_policy = ReclaimPolicy(i);
              found = true;
              break;
            }
          }

          if (!found) {
            print_usage(argv[0]);
            fprintf(stderr, "[Error] argument --r_policy: invalid choice: %s (choose from",
                optarg);
            for (int i = 0; i < n_r_polices; i++) {
              fprintf(stderr, " \'%s\'", r_policies[i]);
            }
            fprintf(stderr, ")\n");
            exit(EXIT_FAILURE);
          }
        }
        break;
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
          this->model_names = model_names;
        }
        pass_model = true;
        break;
      case 'c':
        this->concurrency = (int)strtol(optarg, NULL, 10);
        pass_concurrency = true;
        break;
      case 'd':
        this->dump = std::string(optarg);
        break;
      case 'r':
        this->rate = (int)strtol(optarg, NULL, 10);
        pass_rate = true;
        break;
      case 'p':
        this->mp_size = (int)strtol(optarg, NULL, 10);
        break;
      case 's':
        this->slo = (int)strtol(optarg, NULL, 10);
        break;
      case 'e':
        found = false;
        for (int i = 0; i < n_engine_types; i++) {
          if (!strcmp(engine_types[i], optarg)) {
            this->engine_type = EngineType(i);
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
            this->workload_type = WorkloadType(i);
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

  if (!(pass_model && pass_concurrency && pass_rate)) {
    fprintf(stderr, "[Error] the following arguments are required:");
    if (!pass_model)
      fprintf(stderr, " --model_name/-m");
    if (!pass_concurrency)
      fprintf(stderr, " --concurrency/-c");
    if (!pass_rate)
      fprintf(stderr, " --rate/-r");

    exit(EXIT_FAILURE);
  }
}


void BenchmarkOptions::print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME [--device/-d DEVICES [DEVICES ...]]\n"
      "\t\t[--engine/-e {in_memory,demand,pipeline,deepplan}]\n"
      "\t\t[--batch/-b BATCH_SIZE\n",
      program_name);
}


void BenchmarkOptions::parseOptions(int argc, char** argv) {
  struct option long_options[] =
  {
    {"help",    no_argument,       0, 'h' },
    {"model",   required_argument, 0, 'm' },
    {"engine",  required_argument, 0, 'e' },
    {"devices", required_argument, 0, 'd' },
    {"batch",   required_argument, 0, 'b' },
    {0,         0,                 0,  0  }
  };

  char flag;

  char engine_types[][20] = { "in_memory", "demand", "pipeline", "deepplan"};
  int n_types = sizeof(engine_types) / 20;
  bool found = false;
  bool pass_model = false;

  this->num_warmup  = 20;
  this->num_test    = 200;
  this->batch_size  = 1;
  this->engine_type = EngineType::IN_MEMORY;
  this->devices     = std::vector<int>(1, 0); // = [0]

  while ((flag = getopt_long(argc, argv, "b:d:e:hm:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        this->model_name = std::string(optarg);
        pass_model = true;
        break;
      case 'e':
        found = false;
        for (int i = 0; i < n_types; i++) {
          if (!strcmp(engine_types[i], optarg)) {
            this->engine_type = EngineType(i);
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
        this->batch_size = (int)strtol(optarg, NULL, 10);
        break;
      case 'd':
        optind--;
        {
          std::vector<int> devices;
          for ( ; optind < argc && *argv[optind] != '-'; optind++) {
            devices.push_back((int)strtol(argv[optind], NULL, 10));
          }
          this->devices = devices;
        }
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


void CacheStudyOptions::print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --model/-m MODEL_NAME\n"
      "\t\t[--batch/-b BATCH_SIZE] [--engine/-e {pipeline,deepplan}] [--verbose/-v]\n",
      program_name);
}


void CacheStudyOptions::parseOptions(int argc, char** argv) {
  struct option long_options[] =
  {
    {"verbose", no_argument,       0, 'v' },
    {"help",    no_argument,       0, 'h' },
    {"model",   required_argument, 0, 'm' },
    {"batch",   required_argument, 0, 'b' },
    {"engine",  required_argument, 0, 'e' },
    {0,         0,                 0,  0  }
  };

  char flag;

  bool found = false;
  bool pass_model = false;

  char engine_types[][20] = {"pipeline", "deepplan"};
  int n_types = sizeof(engine_types) / 20;

  this->num_warmup  = 10;
  this->num_test    = 10;
  this->batch_size  = 1;
  this->engine_type = EngineType::PIPESWITCH;
  this->verbose     = false;

  while ((flag = getopt_long(argc, argv, "b:hm:v:e:", long_options, NULL)) != -1) { 
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'm':
        pass_model = true;
        this->model_name = std::string(optarg);
        break;
      case 'b':
        this->batch_size = strtoul(optarg, NULL, 10);
        break;
      case 'v':
        this->verbose = true;
        break;
      case 'e':
        found = false;
        for (int i = 0; i < n_types; i++) {
          if (!strcmp(engine_types[i], optarg)) {
            this->engine_type = EngineType(i+2);
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

  if (!pass_model) {
    print_usage(argv[0]);
    fprintf(stderr, "[Error] the following arguments are required: --model_name/-m\n");
    exit(EXIT_FAILURE);
  }
}
