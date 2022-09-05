#include <iostream>
#include <getopt.h>
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>

#include <util.h>
#include <pcm/cpucounters.h>
#include <pcm/utils.h>

typedef enum {
  EMB,
  LINEAR,
  CONV
} LayerType;

struct PCMPCIeOptions {
  LayerType layer_type;
  int in_features;
  int out_features;
  std::vector<int64_t> input_sizes;
  int kernel_size;
};

static struct option long_options[] =
{
  {"help",          no_argument,        0,  'h' },
  {"layer_type",    required_argument,  0,  't' },
  {"in_features",   required_argument,  0,  'i' },
  {"out_features",  required_argument,  0,  'o' },
  {"input_sizes",   required_argument,  0,  's' },
  {"kernel_size",   required_argument,  0,  'k' },
  {0, 0, 0, 0}
};

int n_warmup = 10;
int n_test = 100;

static void print_usage(char* program_name) {
  fprintf(stderr,
      "Usage : %s [-h] --layer_type/-t {emb, linear, conv}\n"
      "\t\t--in_features/-i IN_FEATURES --out_features/-o OUT_FEATURES\n"
      "\t\t--input_sizes/-s INPUT_SIZES [--kernel_size/-k KERNEL_SIZE]\n",
      program_name);
  exit(EXIT_FAILURE);
}

void parseOptions(PCMPCIeOptions** pcm_pcie_options, int argc, char** argv) {
  *pcm_pcie_options = new PCMPCIeOptions();
  auto options = *pcm_pcie_options;

  char layer_types[][20] = {"emb", "linear", "conv"};
  int n_layer_types = sizeof(layer_types) / 20;
  char flag;
  bool found_type, found_in, found_out, found_input;
  found_type = false;
  found_in = false;
  found_out = false;
  found_input = false;

  options->kernel_size = 1;

  while ((flag = getopt_long(argc, argv, "hi:k:o:s:t:", long_options, NULL)) != -1) {
    switch (flag) {
      case 'h':
        print_usage(argv[0]);
        break;
      case 'i':
        found_in = true;
        options->in_features = (int)strtol(optarg, NULL, 10);
        break;
      case 'k':
        options->kernel_size = (int)strtol(optarg, NULL, 10);
        break;
      case 'o':
        found_out = true;
        options->out_features = (int)strtol(optarg, NULL, 10);
        break;
      case 's':
        found_input = true;
        optind--;
        {
          std::vector<int64_t> sizes;
          for ( ; optind < argc && *argv[optind] != '-'; optind++) {
            sizes.push_back((int64_t)strtoll(argv[optind], NULL, 10));
          }
          options->input_sizes = sizes;
        }
        break;
      case 't':
        for (int i = 0; i < n_layer_types; i++) {
          if (!strcmp(layer_types[i], optarg)) {
            options->layer_type = LayerType(i);
            found_type = true;
            break;
          }
        }
        if (!found_type) {
          fprintf(stderr, "[Error] argument --layer_type/-t: invalid choice: %s\n", optarg);
          print_usage(argv[0]);
        }
        break;
    }
  }

  std::vector<std::string> msg;
  if (!(found_in && found_out && found_type && found_input)) {
    if (!found_in) {
      msg.push_back("--in_features/-i");
    }
    if (!found_out) {
      msg.push_back("--out_features/-o");
    }
    if (!found_type) {
      msg.push_back("--layer_type/-t");
    }
    if (!found_input) {
      msg.push_back("--input_sies/-s");
    }
    std::cerr << "[Error] the following arguments are required: ";
    for (int i = 0; i < msg.size(); i++) {
      std::cerr << msg[i];
      if (i < msg.size()-1) {
        std::cerr << ", ";
      }
    }
    std::cerr << "\n";
    exit(EXIT_FAILURE);
  }

}

struct Linear : torch::nn::Module {
  Linear(int64_t in_features, int64_t out_features)
    : linear(register_module("linear", torch::nn::Linear(in_features, out_features))) {};

  torch::Tensor forward(torch::Tensor x) {
    return linear(x);
  }

  torch::nn::Linear linear;
};

struct Embedding : torch::nn::Module {
  Embedding(int64_t in_features, int64_t out_features)
    : embedding(register_module("embedding", torch::nn::Embedding(in_features, out_features))) {};

  torch::Tensor forward(torch::Tensor x) {
    return embedding(x);
  }

  torch::nn::Embedding embedding;
};


struct Conv2d : torch::nn::Module {
  Conv2d(int64_t in_features, int64_t out_features,
       int64_t kernel_size)
    : conv2d(register_module("conv2d",
              torch::nn::Conv2d(
                in_features, out_features, kernel_size
              )
          )) {};

  torch::Tensor forward(torch::Tensor x) {
    return conv2d(x);
  }

  torch::nn::Conv2d conv2d;
};

torch::nn::Module* getLayer(PCMPCIeOptions* options) {
  LayerType layer_type = options->layer_type;
  int in_features = options->in_features;
  int out_features = options->out_features;
  int kernel_size = options->kernel_size;

  torch::nn::Module* layer;

  switch (layer_type) {
    case EMB:
      layer = new Embedding(in_features, out_features);
      break;
    case LINEAR:
      layer = new Linear(in_features, out_features);
      break;
    case CONV:
      layer = new Conv2d(in_features, out_features, kernel_size);
      break;
  }

  return layer;
}

at::Tensor getInput(PCMPCIeOptions* options) {
  LayerType layer_type = options->layer_type;
  int in_features = options->in_features;
  std::vector<int64_t> input_sizes = options->input_sizes;
  input_sizes.insert(input_sizes.begin(), 1); // batch size = 1;

  at::IntArrayRef sizes = torch::ArrayRef<int64_t>(input_sizes.data(), input_sizes.size());

  auto option = torch::TensorOptions();
  at::Tensor x;
  switch (layer_type) {
    case EMB:
      option = option.dtype(torch::kInt64);
      x = torch::randint(in_features, sizes, option);
      break;
    case LINEAR:
    case CONV:
      option = option.dtype(torch::kFloat32);
      x = torch::randn(sizes, option);
      break;

  }

  return x;
}

int measurePCMPCIe(PCM* m, std::function<void(void)> func, std::function<void(void)> reset) {
  PCIeCounterState* before = new PCIeCounterState[m->getNumSockets()];
  PCIeCounterState* after = new PCIeCounterState[m->getNumSockets()];

  float totalRdCur = 0;
  float numRdCur;
  for (int i = 0; i < n_warmup + n_test; i++) {
    m->programPCIeCounters(m->SKX_RdCur, 0, 0, m->PRQ, 0);

    cudaDeviceSynchronize();
    for (int s = 0; s < m->getNumSockets(); s++) {
      before[s] = m->getPCIeCounterState(s);
    }

    func();

    cudaDeviceSynchronize();
    for (int s = 0; s < m->getNumSockets(); s++) {
      after[s] = m->getPCIeCounterState(s);
    }

    numRdCur = 0;
    for (int s = 0; s < m->getNumSockets(); s++) {
      numRdCur += getNumberOfEvents(before[s], after[s]);
    }

    reset();

    if (i >= n_warmup)
      totalRdCur += numRdCur;
  }

  return (int)(totalRdCur / n_test);
}

double measureTime(std::function<void(void)> func, std::function<void(void)> reset) {
  double t1, t2, total;
  total = 0;

  for (int i = 0; i < n_warmup + n_test; i++) {
    cudaDeviceSynchronize();
    t1 = util::now();

    func();

    cudaDeviceSynchronize();
    t2 = util::now();

    reset();

    if (i >= n_warmup)
      total = (t2-t1);
  }

  return (total / n_test / 1e6);
}

int main(int argc, char** argv) {
  PCMPCIeOptions* options;
  parseOptions(&options, argc, argv);

  PCM* m = PCM::getInstance();
  m->disableJKTWorkaround();

  auto x = getInput(options);
  x = x.to(at::kCUDA);

  auto layer = getLayer(options);

  auto load_func = [&layer]() { layer->to(at::kCUDA); };
  auto exec_func = [&layer, &x]() {
    layer->to(at::kCUDA);
    if (auto emb = dynamic_cast<Embedding*>(layer)) {
      emb->forward(x);
    }
    else if (auto linear = dynamic_cast<Linear*>(layer)) {
      linear->forward(x);
    }
    else if (auto conv2d = dynamic_cast<Conv2d*>(layer)) {
      conv2d->forward(x);
    }
  };
  auto load_reset = [&layer]() { layer->to(at::kCPU); };
  auto dummy_reset = []() {};

  auto numLoadRdCur = measurePCMPCIe(m, load_func, load_reset);
  auto loadTime = measureTime(load_func, load_reset);

  layer->to(at::kCUDA);
  auto execTime = measureTime(exec_func, dummy_reset);


  layer->apply([](torch::nn::Module& module) {
    // Then move every parameter to the new dtype/device.
    for (auto& parameter : module.named_parameters(/*recurse=*/false)) {
      parameter->set_data(torch::autograd::Variable(*parameter).cuda_host());
    }
    // Then move every buffer to the new dtype/device.
    for (auto& buffer : module.named_buffers(/*recurse=*/false)) {
      buffer->set_data(torch::autograd::Variable(*buffer).cuda_host());
    }
  });

  auto numDHARdCur = measurePCMPCIe(m, exec_func, dummy_reset);
  auto execDHATime = measureTime(exec_func, dummy_reset);

  std::cout << "Load numRdCur : " << numLoadRdCur << "\n";
  std::cout << "DHA numRdCur : " << numDHARdCur << "\n";
  std::cout << "Load Time : " << loadTime << " ms\n";
  std::cout << "Exec Time : " << execTime << " ms\n";
  std::cout << "DHA Exec Time : " << execDHATime << " ms\n";

  return 0;
}
