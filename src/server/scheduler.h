#include <util.h>
#include <options.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <server/worker.h>
#include <deepplan/model.h>
#include <optional>
#include <set>
#include <mutex>

// Reference https://gitlab.mpi-sws.org/cld/ml/clockwork/
class WorkerTracker {
 private:
  struct Work { int id; uint64_t exec_time; };
  std::deque<Work> outstandings;

  uint64_t total_outstanding_time = 0UL;
  uint64_t work_begin = 0UL;

 public:
  WorkerTracker() {};

  uint64_t available() {
    return std::max(work_begin + total_outstanding_time, util::now());
  }

  void update(int id, uint64_t end_time) {
    if (outstandings.front().id == id) {
      auto work = outstandings.front();
      total_outstanding_time -= work.exec_time;
      work_begin = end_time;
      outstandings.pop_front();
    }
    else {
      auto it = outstandings.begin();
      for (it; it != outstandings.end(); it++) {
        if (it->id == id) {
          total_outstanding_time -= it->exec_time;
          work_begin += end_time;
          outstandings.erase(it);
        }
      }
    }
  }

  void add_work(int id, uint64_t exec_time) {
    if (outstandings.empty()) {
      work_begin = std::max(work_begin, util::now());
    }
    outstandings.push_back({id, exec_time});
  }
};


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

  WorkerTracker exec;

  std::mutex exec_mutex;

  std::set<InferTask> requests_;

  std::queue<InferTask> timeouts_;

  const ServerOptions& options_;

  std::atomic_int action_seed_id = 0;

  // How far ahead, in nanoseconds, should the scheduler schedule.
  // Default is 10ms.
  uint64_t schedule_ahead = 1e7;
};
