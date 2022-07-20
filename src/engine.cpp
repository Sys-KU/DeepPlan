#include <model.h>
#include <engine.h>
#include <util.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "tbb/concurrent_queue.h"
#include <torch/script.h>

namespace engine {

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
      : modules(modules),
        device(device) {};

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
    queue.abort();
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
    at::cuda::CUDAStreamGuard guard(stream);
    at::Device device("cuda:" + std::to_string(device_));
    std::shared_ptr<Task> task;

    while (!is_finished) {
      try {
        queue.pop(task);
        at::Device target_device("cuda:" + std::to_string(task->device));

        for (auto& module : task->modules) {
          module.synchronize(device);
          module.to_and_record(target_device, true);
        }
      }
      catch (...) {};
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
    at::cuda::CUDAStreamGuard guard(stream);
    at::Device device("cuda:" + std::to_string(device_));
    std::shared_ptr<Task> task;

    while (!is_finished) {
      try {
        queue.pop(task);
        int target_device = task->device;
        for (auto& module : task->modules) {
          module.to_and_record(device, true);

          if (target_device != device_) {
            g_nvlink_thrs[device_]->transfer_modules(module, target_device);
          }
        }
      }
      catch (...) {};
    }
  }
};

static void init(void) {
  n_device = torch::cuda::device_count();

  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs.push_back(std::move(new PCIeThread(i)));
    g_nvlink_thrs.push_back(std::move(new NVLinkThread(i)));
    g_exec_streams.push_back(std::move(c10::cuda::getStreamFromPool(false, i)));
  }
}

static void deinit(void) {
  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i]->stop();
    g_nvlink_thrs[i]->stop();
  }
}

namespace {
  struct Cleaner {
    Cleaner() {init();}
    ~Cleaner() {deinit();}
  };

  Cleaner cleaner;
}

class PipelineEngine : public Engine {
 public:
  PipelineEngine()
    : Engine() {};

  void run(Model* model, ScriptModuleInput& x) {
    int target_device = model->devices[0];

    assert(n_device > target_device);

    at::cuda::CUDAStreamGuard guard(g_exec_streams[target_device]);
    if (!model->is_cuda) {
      for (auto const& it : model->device_map) {
        std::vector<ScriptModule> modules;
        for (auto idx : it.second) {
          modules.push_back(model->layers[idx]);
        }
        g_pcie_thrs[it.first]->transfer_modules(modules, target_device);
      }
    }
    model->model.forward(x);
    model->is_cuda = true;
    return;
  }
};

PipelineEngine engine;

void run(Model* model, ScriptModuleInput& x) {
  c10::InferenceMode guard;
  engine.run(model, x);
}

}
