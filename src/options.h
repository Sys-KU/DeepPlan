#pragma once

#include <util.h>
#include <vector>

struct Options {
 public:
  virtual void print_usage(char* program_name) = 0;
  virtual void parseOptions(int argc, char** argv) = 0;
};

struct ServerOptions : public Options{
 public:
  float watermark;

  void print_usage(char* program_name);
  void parseOptions(int argc, char** argv);
};

struct ClientOptions : public Options{
 public:
  enum class WorkloadType {
    SIMPLE_UNIFORM = 0,
    SIMPLE_ZIPF,
    BURSTY,
    AZURE,
    SKEW,
  } workload_type;

  std::vector<std::string> model_names;
  int concurrency;
  int rate;
  int mp_size;
  EngineType engine_type;
  ReclaimPolicy r_policy;
  int slo;
  std::string dump;
  int n_warmup;
  int n_test;

  void print_usage(char* program_name);
  void parseOptions(int argc, char** argv);
};

struct BenchmarkOptions {
 public:
  std::string model_name;
  EngineType engine_type;
  std::vector<int> devices;
  int batch_size;
  int num_warmup;
  int num_test;

  void print_usage(char* program_name);
  void parseOptions(int argc, char** argv);
};

struct CacheStudyOptions {
 public:
  std::string model_name;
  bool verbose;
  int batch_size;
  int num_warmup;
  int num_test;

  void print_usage(char* program_name);
  void parseOptions(int argc, char** argv);
};
