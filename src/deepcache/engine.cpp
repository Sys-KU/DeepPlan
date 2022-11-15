#include <deepcache/model.h>
#include <deepcache/engine.h>
#include <util.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "tbb/concurrent_queue.h"
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace deepcache {

class PCIeThread;

static std::vector<c10::cuda::CUDAStream> g_exec_streams;
static std::vector<PCIeThread*> g_pcie_thrs;
static int n_device;

class LoadThread {
 public:
  LoadThread(int device)
   : device_(device),
     is_finished(false),
     stream(c10::cuda::getStreamFromPool(false, device)) {};

  struct Task {
   public:
    Task(std::vector<ScriptModule> modules, int device)
      : type(Type::request),
        modules(modules),
        device(device) {};

    Task()
      : type(Type::end) {};

    enum class Type {
      request = 0,
      end
    } type;

    std::vector<ScriptModule> modules;
    int device;
  };

  void transfer_modules(std::vector<ScriptModule>& modules, int target_device) {
    if (!modules.empty())
      queue.push(std::make_shared<Task>(modules, target_device));
  }

  virtual void init() = 0;

  virtual void Loop() = 0;

  void stop() {
    is_finished = true;
    queue.push(std::make_shared<Task>()); // Insert EndofItem
    if (thr.joinable())
      thr.join();
  }

 protected:
  tbb::concurrent_bounded_queue<std::shared_ptr<Task>> queue;
  c10::cuda::CUDAStream stream;
  std::thread thr;
  std::atomic<bool> is_finished;
  int device_;
};

class PCIeThread : public LoadThread {
 public:
  PCIeThread(int device)
   : LoadThread(device) { init(); };

  void init() {
    thr = std::thread(std::bind(&PCIeThread::Loop, this));
  }

  void Loop() {
    at::Device device(at::kCUDA, device_);
    at::cuda::CUDAStreamGuard guard(stream);
    c10::cuda::CUDAGuard device_guard(device);

    std::shared_ptr<Task> task;

    while (!is_finished) {
      queue.pop(task);
      if (task->type == Task::Type::end) {
        break;
      }

      int target_device = task->device;

      for (auto& module : task->modules) {
        module.to_and_record(device, true);
      }
    }
  }
};

void Init(void) {
  n_device = torch::cuda::device_count();
  torch::jit::getBailoutDepth() = 0;

  g_pcie_thrs.resize(n_device);

  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i] = new PCIeThread(i);
    g_exec_streams.push_back(std::move(c10::cuda::getStreamFromPool(false, i)));
  }
}

void Deinit(void) {
  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i]->stop();
  }
}

class PipelineEngine : public Engine {
 public:
  PipelineEngine()
    : Engine() {};

  torch::jit::IValue run(Model* model, ScriptModuleInput& x) {
    int target_device = model->target_device.index();
    torch::jit::IValue outputs;

    assert(n_device > target_device);

    LoadLayers(model);

    {
      at::cuda::CUDAStreamGuard stream_guard(g_exec_streams[target_device]);
      outputs = model->model.forward(x);
    }

    return outputs;
  }
};

static PipelineEngine engine;

torch::jit::IValue RunEngine(Model* model, ScriptModuleInput& x) {
  c10::cuda::CUDAGuard device_guard(model->target_device);
  auto outputs = engine.run(model, x);

  return outputs;
}

void LoadLayers(Model* model) {
  int target_device = model->target_device.index();

  auto host_layers = model->get_host_layers();
  if (!host_layers.empty()) {
    std::vector<ScriptModule> modules;
    for (auto l : host_layers) {
      modules.push_back(model->layers[l]);
    }

    g_pcie_thrs[target_device]->transfer_modules(modules, target_device);
  }

  // FIXME
  for (auto l : host_layers) {
    model->layers_load_info[l] = Device::CUDA;
  }
  model->is_cuda = true;
  model->remained_size = 0;
}

}
