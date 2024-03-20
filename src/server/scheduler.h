#include <util.h>
#include <options.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <server/worker.h>
#include <deepplan/model.h>
#include <optional>
#include <set>

class Scheduler {
 public:
  Scheduler(int device, const ServerOptions& options,
            ModelPool* model_pool, std::string name="");

  void enqueue_request(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb,
      std::function<void(serverapi::TimeoutResponse*)> timeout_cb);

  void handle_requests();

  void handle_timeouts();

  void clear_models();

  void set_r_policy(ReclaimPolicy r_policy);

  void sync_setup();

  void stop();

  at::Device device;

  std::string name;

  ModelPool* model_pool;

 private:
  Worker* worker_;

  std::set<InferTask> requests_;

  std::queue<InferTask> timeouts_;

  const ServerOptions& options_;
};
