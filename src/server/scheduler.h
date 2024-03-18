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
  Scheduler(int device, const ServerOptions& options, std::string name="");

  void enqueue_request(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb);

  void handle_requests();

  void clear_models();

  void add_models(std::vector<std::string> model_names, int n_models,
                  EngineType engine_type, int slo_ms, std::vector<int> devices);

  void reset_models(std::vector<std::string> model_names, int n_models,
                    EngineType engine_type, int slo_ms, std::vector<int> devices);

  void set_r_policy(ReclaimPolicy r_policy);

  void stop();

  at::Device device;

  std::string name;

 private:
  Worker* worker_;

  std::set<InferTask> requests_;

  const ServerOptions& options_;
};
