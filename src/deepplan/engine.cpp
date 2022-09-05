#include <deepplan/model.h>
#include <deepplan/engine.h>
#include <util.h>

#include <cassert>
#include <future>
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
        device(device),
        cb(cb) {};

    Task(std::function<void(void)> cb, int device)
      : type(Type::dummy),
        cb(cb),
        device(device) {};

    Task()
      : type(Type::end) {};

    enum class Type {
      request = 0,
      dummy,
      end
    } type;

    std::vector<ScriptModule> modules;
    std::function<void(void)> cb;
    int device;
  };

  void transfer_modules(std::vector<ScriptModule>& modules, int target_device) {
    if (!modules.empty())
      queue.push(std::make_shared<Task>(modules, target_device));
  }

  void transfer_module(ScriptModule module, int target_device) {
    std::vector<ScriptModule> modules;
    modules.push_back(std::move(module));
    queue.push(std::make_shared<Task>(modules, target_device));
  }

  void transfer_dummy(std::function<void(void)> cb, int target_device) {
    queue.push(std::make_shared<Task>(cb, target_device));
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
      else if (task->type == Task::Type::request) {
        at::Device target_device(at::kCUDA, task->device);

        for (auto& module : task->modules) {
          module.synchronize(device);
          module.to_and_record(target_device, true);
        }
      }
      else {
        task->cb();
      }
    }
  }

};

class PCIeThread : public LoadThread {
 public:
  PCIeThread(int device, bool pipeline_transmission)
   : pipeline_transmission(pipeline_transmission),
     LoadThread(device) { init(); };

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
      else if (task->type == Task::Type::request) {
        int target_device = task->device;
        bool use_pt = target_device != device_;

        for (auto& module : task->modules) {
          module.to_and_record(device, true);

          if (use_pt) {
            if (pipeline_transmission)
              g_nvlink_thrs[device_]->transfer_module(module, target_device);
          }
        }
        if (use_pt) {
          if (!pipeline_transmission)
            g_nvlink_thrs[device_]->transfer_modules(task->modules, target_device);
        }
      }
      else {
        bool use_pt = task->device != device_;
        if (use_pt)
          g_nvlink_thrs[device_]->transfer_dummy(task->cb, task->device);
        else
          task->cb();
      }
    }
  }

  bool pipeline_transmission;
};

void Init(bool pipeline_transmission) {
  n_device = torch::cuda::device_count();
  torch::jit::getBailoutDepth() = 0;

  g_pcie_thrs.resize(n_device);
  g_nvlink_thrs.resize(n_device);

  for (int i = 0; i < n_device; i++) {
    g_pcie_thrs[i] = new PCIeThread(i, pipeline_transmission);
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

  void load(Model* model) {
    int target_device = model->target_device.index();

    assert(n_device > target_device);

    if (!model->is_cuda) {
      auto promises =
        std::vector<std::promise<cudaEvent_t>>(model->devices.size());
      for (int i = 0; i < model->devices.size(); i++) {
        int device = model->devices[i];
        auto cb = [this, &promises, i]() {
          cudaEvent_t event;
          cudaEventCreate(&event);
          cudaEventRecord(event, c10::cuda::getCurrentCUDAStream());
          promises[i].set_value(event);
        };

        std::vector<ScriptModule> modules;
        for (auto idx : model->device_map[device]) {
          modules.push_back(model->layers[idx]);
        }
        g_pcie_thrs[device]->transfer_modules(modules, target_device);
        g_pcie_thrs[device]->transfer_dummy(cb, target_device);
      }

      for (int i = 0; i < model->devices.size(); i++) {
        auto event = promises[i].get_future().get();
      }
    }

    model->is_cuda = true;
  }
};

static PipelineEngine engine;

torch::jit::IValue RunEngine(Model* model, ScriptModuleInput& x) {
  c10::cuda::CUDAGuard device_guard(model->target_device);
  auto outputs = engine.run(model, x);

  return outputs;
}

void Load(Model* model) {
  c10::cuda::CUDAGuard device_guard(model->target_device);
  engine.load(model);
}

}
