#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "tbb/concurrent_queue.h"
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace deepplan {

class PCIeThread;
class NVLinkThread;

static std::vector<c10::cuda::CUDAStream> g_exec_streams;
static std::vector<PCIeThread*> g_pcie_thrs;
static std::vector<NVLinkThread*> g_nvlink_thrs;
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

class NVLinkThread : public LoadThread {
 public:
  NVLinkThread(int device)
    : LoadThread(device) { init(); };

  void init() {
    thr = std::thread(std::bind(&NVLinkThread::Loop, this));
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
      at::Device target_device(at::kCUDA, task->device);

      for (auto& module : task->modules) {
        module.synchronize(device);
        module.to_and_record(target_device, true);
      }
    }
  }

  void transfer_modules(ScriptModule module, int target_device) {
    std::vector<ScriptModule> modules;
    modules.push_back(std::move(module));
    queue.push(std::make_shared<Task>(modules, target_device));
  }
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

        if (target_device != device_) {
          g_nvlink_thrs[device_]->transfer_modules(module, target_device);
        }
      }
    }
  }
};

void Init(void) {
  n_device = torch::cuda::device_count();
  torch::jit::getBailoutDepth() = 0;

  g_pcie_thrs.resize(n_device);
  g_nvlink_thrs.resize(n_device);

  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i] = new PCIeThread(i);
    g_nvlink_thrs[i] = new NVLinkThread(i);
    g_exec_streams.push_back(std::move(c10::cuda::getStreamFromPool(false, i)));
  }
}

void Deinit(void) {
  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i]->stop();
    g_nvlink_thrs[i]->stop();
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

    if (!model->is_cuda) {

      for (int device : model->devices) {
        std::vector<ScriptModule> modules;
        for (auto idx : model->device_map[device]) {
          modules.push_back(model->layers[idx]);
        }
        g_pcie_thrs[device]->transfer_modules(modules, target_device);
      }
    }

    {
      at::cuda::CUDAStreamGuard stream_guard(g_exec_streams[target_device]);
      outputs = model->model.forward(x);
    }
    model->is_cuda = true;

    return outputs;
  }
};

static PipelineEngine engine;

torch::jit::IValue RunEngine(Model* model, ScriptModuleInput& x) {
  c10::cuda::CUDAGuard device_guard(model->target_device);
  auto outputs = engine.run(model, x);

  return outputs;
}

}
